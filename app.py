"""
ML Experiment Agent — combined version

Includes:
  - Classification AND regression support (auto-detected from target column)
  - Friendly validation errors instead of raw crashes on bad CSVs/targets
  - Session persistence: experiment log survives a browser refresh (local JSON file)
  - Lightweight hyperparameter search: agent can propose a couple of param
    options per experiment; the app tries each and logs the best
  - Explainability: feature importances / coefficients shown for the best model
  - Time-budget enforcement: reactive loop respects a user-set seconds budget
  - Downloadable markdown report of the full run
  - Reactive agent loop (plan one step at a time, react to real results) AND
    batch mode (plan everything upfront) — both call Groq (Llama via Groq)
  - Manual single-model fallback that needs no API key

Run with: streamlit run app.py
"""

import json
import os
import time
import warnings

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="ML Experiment Agent", layout="wide")

GROQ_MODEL = "openai/gpt-oss-120b"
LOG_FILE = "experiment_log.json"

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}
REGRESSION_MODELS = {
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
}
ALL_MODELS = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}

# ---------------------------------------------------------------------------
# Session state (+ load persisted log from disk, if any)
# ---------------------------------------------------------------------------
def _load_persisted_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


for key, default in [
    ("experiment_log", None),  # None sentinel so we only load-from-disk once
    ("df", None),
    ("plan", None),
    ("summary", None),
    ("groq_key_input", ""),
    ("best_model_artifact", None),  # (model, feature_names, task_type) for explainability
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.experiment_log is None:
    st.session_state.experiment_log = _load_persisted_log()


def _persist_log():
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.experiment_log, f)
    except Exception:
        pass  # best-effort; app still works without persistence


def log_experiment(model_name, params, metrics, task_type, rationale="", notes=""):
    st.session_state.experiment_log.append(
        {
            "run": len(st.session_state.experiment_log) + 1,
            "task_type": task_type,
            "model": model_name,
            "params": json.dumps(params),
            "rationale": rationale,
            "notes": notes,
            **metrics,
        }
    )
    _persist_log()


def primary_metric_name(task_type):
    return "f1" if task_type == "classification" else "r2"


def best_row(log_df):
    """Pick the best row, handling mixed classification/regression logs gracefully."""
    metric = None
    if "f1" in log_df.columns and log_df["f1"].notna().any():
        metric = "f1"
    elif "r2" in log_df.columns and log_df["r2"].notna().any():
        metric = "r2"
    if metric is None:
        return None, None
    idx = log_df[metric].idxmax()
    return log_df.loc[idx], metric


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = st.session_state.get("groq_key_input") or None
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Dataset handling: task detection + validation + summary
# ---------------------------------------------------------------------------
def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    y = df[target_col]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"


def validate_dataset(df: pd.DataFrame, target_col: str, task_type: str):
    """Raise ValueError with a friendly message if the dataset can't be used."""
    if df is None or df.empty:
        raise ValueError("The uploaded CSV appears to be empty.")
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' was not found in the dataset.")
    if df[target_col].isna().all():
        raise ValueError(f"Target column '{target_col}' has no non-missing values.")

    working = df.dropna()
    if working.shape[0] < 10:
        raise ValueError(
            f"After removing rows with missing values, only {working.shape[0]} rows remain "
            "— not enough to train/test a model reliably (need at least ~10)."
        )
    if task_type == "classification":
        n_classes = working[target_col].nunique()
        if n_classes < 2:
            raise ValueError(
                f"Target column '{target_col}' only has {n_classes} distinct value(s) — "
                "classification needs at least 2 classes."
            )
        smallest_class_count = working[target_col].value_counts().min()
        if smallest_class_count < 2:
            raise ValueError(
                f"The smallest class in '{target_col}' has only {smallest_class_count} example(s) "
                "— add more data or merge rare classes before training."
            )


def summarize_dataset(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    schema = []
    for col in df.columns:
        schema.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique()),
            }
        )
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": schema,
        "target_column": target_col,
        "task_type": task_type,
    }
    if task_type == "classification":
        summary["target_class_balance"] = (
            df[target_col].value_counts(normalize=True).round(3).to_dict()
        )
    else:
        summary["target_stats"] = {
            "mean": round(float(df[target_col].mean()), 4),
            "std": round(float(df[target_col].std()), 4),
            "min": round(float(df[target_col].min()), 4),
            "max": round(float(df[target_col].max()), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Execution: real sklearn training, with lightweight param search + explainability
# ---------------------------------------------------------------------------
def _prep_xy(df, target_col, model_key):
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if model_key in ("logistic_regression",) and (y.dtype == "object" or y.dtype.name == "category"):
        y = LabelEncoder().fit_transform(y.astype(str))
    elif model_key in CLASSIFICATION_MODELS and (y.dtype == "object" or y.dtype.name == "category"):
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def _score(task_type, y_test, preds):
    if task_type == "classification":
        return {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds, average="weighted"), 4),
            "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average="weighted"), 4),
        }
    else:
        return {
            "mae": round(mean_absolute_error(y_test, preds), 4),
            "r2": round(r2_score(y_test, preds), 4),
        }


def _train_once(df, target_col, model_key, params, task_type):
    X, y = _prep_xy(df, target_col, model_key)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    needs_scaling = model_key in ("logistic_regression", "linear_regression")
    if needs_scaling:
        scaler = StandardScaler()
        X_train_fit, X_test_fit = scaler.fit_transform(X_train), scaler.transform(X_test)
    else:
        X_train_fit, X_test_fit = X_train, X_test

    model_cls = ALL_MODELS[model_key]
    safe_params = dict(params or {})
    if model_key not in ("logistic_regression", "linear_regression"):
        safe_params.setdefault("random_state", 42)
    model = model_cls(**safe_params)
    model.fit(X_train_fit, y_train)
    preds = model.predict(X_test_fit)
    metrics = _score(task_type, y_test, preds)
    return metrics, model, list(X.columns)


def run_experiment(df, target_col, model_key, params_or_options, task_type):
    """
    params_or_options: either a single params dict, or a list of param dicts
    (a tiny internal hyperparameter search — tries each, keeps the best).
    Returns (best_metrics, chosen_params, model, feature_names).
    """
    if isinstance(params_or_options, list) and params_or_options:
        candidates = params_or_options
    else:
        candidates = [params_or_options or {}]

    metric_key = "f1" if task_type == "classification" else "r2"
    best = None
    for params in candidates:
        try:
            metrics, model, feat_names = _train_once(df, target_col, model_key, params, task_type)
        except Exception:
            continue
        if best is None or metrics.get(metric_key, -1e9) > best[0].get(metric_key, -1e9):
            best = (metrics, params, model, feat_names)

    if best is None:
        raise ValueError(f"All parameter attempts for '{model_key}' failed to train.")
    return best  # (metrics, chosen_params, model, feature_names)


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda p: -p[1])
        return pd.DataFrame(pairs, columns=["feature", "importance"])
    if hasattr(model, "coef_"):
        coefs = model.coef_
        coefs = coefs[0] if getattr(coefs, "ndim", 1) > 1 else coefs
        pairs = sorted(zip(feature_names, coefs), key=lambda p: -abs(p[1]))
        return pd.DataFrame(pairs, columns=["feature", "coefficient"])
    return None


# ---------------------------------------------------------------------------
# Groq prompts
# ---------------------------------------------------------------------------
MODEL_MENU_HINT = (
    'For classification use one of: "logistic_regression", "random_forest", "gradient_boosting". '
    'For regression use one of: "linear_regression", "random_forest_regressor", "gradient_boosting_regressor".'
)


def _valid_model_for_task(model_name, task_type):
    if task_type == "classification":
        return model_name in CLASSIFICATION_MODELS
    return model_name in REGRESSION_MODELS


def call_groq_first_experiment(client, problem_desc, dataset_summary, task_type):
    system_prompt = (
        "You are an ML experiment planning assistant. Given a problem description and a "
        "dataset schema, propose ONE sensible starting experiment for a "
        f"{task_type} task. {MODEL_MENU_HINT} "
        "You may optionally propose 2 slightly different param sets to try (a tiny internal "
        "hyperparameter search) via 'params_options' instead of a single 'params'. "
        "Respond with ONLY a JSON object, no prose, no markdown fences, shaped exactly like:\n"
        '{"model": "...", "params": {...}} OR {"model": "...", "params_options": [{...}, {...}]}, '
        'plus "rationale": "one sentence why". '
        "Keep params simple and safe (n_estimators <= 200, max_depth <= 10)."
    )
    user_prompt = f"Problem: {problem_desc}\n\nDataset summary:\n{json.dumps(dataset_summary, indent=2)}"
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    exp = json.loads(resp.choices[0].message.content)
    return exp if _valid_model_for_task(exp.get("model"), task_type) else None


def call_groq_next_step(client, problem_desc, dataset_summary, log_so_far, task_type):
    system_prompt = (
        "You are an ML experiment agent running an iterative search for a "
        f"{task_type} task. Given results so far, decide ONE of two actions:\n"
        '  - "continue": propose exactly one NEW experiment reacting to what you have '
        "seen (e.g. if a linear model underperformed, try a non-linear one). You may "
        "optionally include 'params_options' (a list of 2 param dicts) instead of a "
        "single 'params' to try a couple of variants.\n"
        '  - "stop": if further experiments are unlikely to help.\n'
        f"{MODEL_MENU_HINT}\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences, shaped exactly:\n"
        '{"action": "continue" | "stop", "next_experiment": {"model": "...", '
        '"params": {...} or "params_options": [...], "rationale": "..."} or null, '
        '"reason": "one sentence"}'
    )
    user_prompt = (
        f"Problem: {problem_desc}\n\nDataset summary:\n{json.dumps(dataset_summary, indent=2)}\n\n"
        f"Experiments so far:\n{json.dumps(log_so_far, indent=2)}"
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    decision = json.loads(resp.choices[0].message.content)
    next_exp = decision.get("next_experiment")
    if decision.get("action") == "continue" and (
        not next_exp or not _valid_model_for_task(next_exp.get("model"), task_type)
    ):
        decision["action"] = "stop"
        decision["reason"] = decision.get("reason", "") + " (fallback: invalid next-step model)"
    return decision


def call_groq_plan(client, problem_desc, dataset_summary, task_type, max_experiments=3):
    system_prompt = (
        "You are an ML experiment planning assistant. Given a problem description and a "
        f"dataset schema, propose a short sequence of experiments for a {task_type} task. "
        f"{MODEL_MENU_HINT} Respond with ONLY a JSON object, no prose, no markdown fences:\n"
        '{"experiments": [{"model": "...", "params": {...}, "rationale": "..."}]}\n'
        f"Propose at most {max_experiments} experiments. Keep params simple and safe."
    )
    user_prompt = f"Problem: {problem_desc}\n\nDataset summary:\n{json.dumps(dataset_summary, indent=2)}"
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    parsed = json.loads(resp.choices[0].message.content)
    experiments = parsed.get("experiments", [])[:max_experiments]
    return [e for e in experiments if _valid_model_for_task(e.get("model"), task_type)]


def call_groq_summary(client, problem_desc, log_rows):
    system_prompt = (
        "You are an ML experiment analyst. Given a problem description and a list of "
        "completed experiment results (classification and/or regression), write a short "
        "(4-6 sentence) plain-English summary: which run performed best, why, and one "
        "concrete suggestion for what to try next. No markdown headers."
    )
    user_prompt = f"Problem: {problem_desc}\n\nResults:\n{json.dumps(log_rows, indent=2)}"
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content


def build_markdown_report(problem_desc, dataset_summary, log_rows, summary_text):
    lines = [
        "# ML Experiment Agent — Report",
        "",
        f"**Problem / goal:** {problem_desc or '(not specified)'}",
        f"**Task type:** {dataset_summary.get('task_type', 'unknown')}",
        f"**Dataset:** {dataset_summary.get('n_rows')} rows × {dataset_summary.get('n_cols')} columns, "
        f"target = `{dataset_summary.get('target_column')}`",
        "",
        "## Experiment log",
        "",
    ]
    if log_rows:
        cols = list(log_rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in log_rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    else:
        lines.append("_No experiments run._")
    lines += ["", "## Agent's comparison", "", summary_text or "_Not generated yet._"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧪 ML Experiment Agent")
st.caption("Give it a problem, a dataset, and a constraint — it plans, runs, tracks, and compares experiments.")

with st.sidebar:
    st.subheader("Groq API Key")
    has_secret = False
    try:
        has_secret = bool(st.secrets["GROQ_API_KEY"])
    except Exception:
        has_secret = False
    if has_secret:
        st.success("Using key from secrets.toml")
    else:
        st.session_state.groq_key_input = st.text_input(
            "Paste your Groq API key",
            type="password",
            value=st.session_state.groq_key_input,
            help="Get one free at console.groq.com. Kept only in this session, never saved to disk.",
        )
    st.caption("Needed for AI-planned experiments. Manual mode below works without it.")
    st.divider()
    time_budget = st.number_input(
        "Time budget for reactive loop (seconds, 0 = no limit)", min_value=0, value=0, step=10
    )

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Upload dataset")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
    if st.session_state.df is not None:
        st.write(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
        st.dataframe(st.session_state.df.head(), width="stretch")

with col_right:
    st.subheader("2. Describe the problem")
    problem_desc = st.text_area(
        "Problem / goal",
        placeholder="e.g. Predict customer churn. Optimize for F1. Keep training under a few minutes.",
        height=100,
    )
    task_type = None
    target_col = None
    if st.session_state.df is not None:
        target_col = st.selectbox("Target column", st.session_state.df.columns)
        task_type = detect_task_type(st.session_state.df, target_col)
        st.caption(f"Detected task type: **{task_type}** (auto-detected from the target column)")
    else:
        st.info("Upload a CSV to select a target column.")

# Validate as early as possible, show a friendly message instead of crashing later
validation_error = None
if st.session_state.df is not None and target_col is not None:
    try:
        validate_dataset(st.session_state.df, target_col, task_type)
    except ValueError as e:
        validation_error = str(e)
        st.error(f"Can't train on this dataset yet: {validation_error}")

st.divider()

st.subheader("3. Agent-planned experiments")
client = get_groq_client()
plan_disabled = (
    st.session_state.df is None or target_col is None or client is None or validation_error is not None
)
if client is None and st.session_state.df is not None:
    st.warning("Add a Groq API key in the sidebar to enable AI planning.")

st.caption(
    "Reactive mode: the agent runs one experiment, looks at the real result, and decides "
    "what to try next. Time budget above (if set) is enforced here."
)
max_reactive_rounds = st.slider("Max experiments (reactive mode)", 2, 6, 4)

if st.button("🔁 Run reactive agent loop", type="primary", disabled=plan_disabled):
    dataset_summary = summarize_dataset(st.session_state.df, target_col, task_type)

    with st.spinner("Agent is proposing a starting experiment..."):
        first_exp = call_groq_first_experiment(client, problem_desc, dataset_summary, task_type)

    if not first_exp:
        st.error("The agent couldn't propose a starting experiment. Try rephrasing the problem.")
    else:
        current_exp = first_exp
        round_num = 0
        loop_start = time.time()
        status_area = st.container()
        stopped_for_budget = False

        while current_exp and round_num < max_reactive_rounds:
            if time_budget and (time.time() - loop_start) > time_budget:
                stopped_for_budget = True
                break

            round_num += 1
            params_or_options = current_exp.get("params_options") or current_exp.get("params", {})
            with status_area:
                st.write(
                    f"**Round {round_num}:** `{current_exp['model']}` {params_or_options} "
                    f"— {current_exp.get('rationale', '')}"
                )
            with st.spinner(f"Running round {round_num}..."):
                start = time.time()
                try:
                    metrics, chosen_params, model, feat_names = run_experiment(
                        st.session_state.df, target_col, current_exp["model"], params_or_options, task_type
                    )
                except ValueError as e:
                    with status_area:
                        st.error(f"Round {round_num} failed to train: {e}")
                    break
                elapsed = round(time.time() - start, 2)
                log_experiment(
                    model_name=current_exp["model"],
                    params=chosen_params,
                    metrics=metrics,
                    task_type=task_type,
                    rationale=current_exp.get("rationale", ""),
                    notes=f"reactive round {round_num}, trained in {elapsed}s",
                )
                metric_key = "f1" if task_type == "classification" else "r2"
                current_best = st.session_state.best_model_artifact
                if current_best is None or metrics[metric_key] > current_best[3]:
                    st.session_state.best_model_artifact = (model, feat_names, task_type, metrics[metric_key])
            with status_area:
                st.write(f"→ {metrics}")

            if round_num >= max_reactive_rounds:
                break
            if time_budget and (time.time() - loop_start) > time_budget:
                stopped_for_budget = True
                break

            with st.spinner("Agent is deciding the next step..."):
                decision = call_groq_next_step(
                    client, problem_desc, dataset_summary, st.session_state.experiment_log, task_type
                )
            if decision.get("action") == "stop":
                with status_area:
                    st.info(f"Agent decided to stop: {decision.get('reason', '')}")
                current_exp = None
            else:
                current_exp = decision.get("next_experiment")

        if stopped_for_budget:
            st.warning(f"Stopped early — time budget of {time_budget}s reached.")

        with st.spinner("Agent is writing the final comparison..."):
            st.session_state.summary = call_groq_summary(
                client, problem_desc, st.session_state.experiment_log
            )
        st.success(f"Done after {round_num} round(s) — see the log and summary below.")

st.divider()
st.caption("Or: batch mode — agent plans a fixed set of experiments upfront, then runs all of them.")

if st.button("🧠 Plan + run experiments (batch)", disabled=plan_disabled):
    dataset_summary = summarize_dataset(st.session_state.df, target_col, task_type)
    with st.spinner("Agent is planning experiments..."):
        st.session_state.plan = call_groq_plan(client, problem_desc, dataset_summary, task_type)

    if not st.session_state.plan:
        st.error("The agent didn't return a usable plan. Try rephrasing the problem description.")
    else:
        st.write("**Plan:**")
        for i, exp in enumerate(st.session_state.plan, 1):
            st.write(f"{i}. `{exp['model']}` {exp.get('params', exp.get('params_options', {}))} — {exp.get('rationale', '')}")

        progress = st.progress(0.0, text="Running experiments...")
        for i, exp in enumerate(st.session_state.plan):
            params_or_options = exp.get("params_options") or exp.get("params", {})
            start = time.time()
            try:
                metrics, chosen_params, model, feat_names = run_experiment(
                    st.session_state.df, target_col, exp["model"], params_or_options, task_type
                )
            except ValueError as e:
                st.error(f"Experiment {i + 1} failed: {e}")
                continue
            elapsed = round(time.time() - start, 2)
            log_experiment(
                model_name=exp["model"],
                params=chosen_params,
                metrics=metrics,
                task_type=task_type,
                rationale=exp.get("rationale", ""),
                notes=f"AI-planned (batch), trained in {elapsed}s",
            )
            metric_key = "f1" if task_type == "classification" else "r2"
            current_best = st.session_state.best_model_artifact
            if current_best is None or metrics[metric_key] > current_best[3]:
                st.session_state.best_model_artifact = (model, feat_names, task_type, metrics[metric_key])
            progress.progress((i + 1) / len(st.session_state.plan), text=f"Ran {i + 1}/{len(st.session_state.plan)}")

        with st.spinner("Agent is comparing results..."):
            st.session_state.summary = call_groq_summary(client, problem_desc, st.session_state.experiment_log)
        st.success("Done — see the log and summary below.")

with st.expander("Or run a single model manually (no API key needed)"):
    manual_options = list(CLASSIFICATION_MODELS.keys()) if task_type != "regression" else list(REGRESSION_MODELS.keys())
    manual_model = st.radio("Model", manual_options, horizontal=True)
    manual_disabled = st.session_state.df is None or target_col is None or validation_error is not None
    if st.button("▶ Run manual experiment", disabled=manual_disabled):
        with st.spinner("Training..."):
            start = time.time()
            metrics, chosen_params, model, feat_names = run_experiment(
                st.session_state.df, target_col, manual_model, {}, task_type
            )
            elapsed = round(time.time() - start, 2)
            log_experiment(
                model_name=manual_model,
                params=chosen_params,
                metrics=metrics,
                task_type=task_type,
                rationale="Manually selected",
                notes=f"trained in {elapsed}s",
            )
            metric_key = "f1" if task_type == "classification" else "r2"
            current_best = st.session_state.best_model_artifact
            if current_best is None or metrics[metric_key] > current_best[3]:
                st.session_state.best_model_artifact = (model, feat_names, task_type, metrics[metric_key])
        st.success(f"Done in {elapsed}s")

st.divider()
st.subheader("4. Experiment log")

if st.session_state.experiment_log:
    log_df = pd.DataFrame(st.session_state.experiment_log)
    st.dataframe(log_df, width="stretch")

    row, metric = best_row(log_df)
    if row is not None:
        st.success(f"Best so far: **{row['model']}** (Run {row['run']}) — {metric.upper()} = {row[metric]}")

    if st.session_state.best_model_artifact:
        model, feat_names, best_task_type, _ = st.session_state.best_model_artifact
        importance_df = get_feature_importance(model, feat_names)
        if importance_df is not None:
            st.write("**What drove the best model's predictions:**")
            st.dataframe(importance_df, width="stretch")

    if st.session_state.summary:
        st.subheader("Agent's comparison")
        st.write(st.session_state.summary)

        report_md = build_markdown_report(
            problem_desc,
            summarize_dataset(st.session_state.df, target_col, task_type) if st.session_state.df is not None else {},
            st.session_state.experiment_log,
            st.session_state.summary,
        )
        st.download_button(
            "⬇ Download report (Markdown)",
            data=report_md,
            file_name="ml_experiment_report.md",
            mime="text/markdown",
        )
else:
    st.info("No experiments run yet. Upload a dataset and hit a Run button above.")

if st.button("Clear log"):
    st.session_state.experiment_log = []
    st.session_state.plan = None
    st.session_state.summary = None
    st.session_state.best_model_artifact = None
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    st.rerun()
