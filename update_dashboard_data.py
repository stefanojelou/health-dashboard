"""
Weekly Dashboard Pipeline
-------------------------
Runs query extraction + merge + validation + git sync in one command:

    python update_dashboard_data.py

Prerequisites:
1) Fill in .env (copy from .env.template)
2) pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "merged_weekly_data.csv"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
QUERIES_DIR = PROJECT_ROOT / "queries"


MYSQL_QUERY_JOBS = {
    "signups": {
        "query_file": "signups_weekly.sql",
        "output_file": "signups_weekly.csv",
        "env_prefix": "MYSQL_CHATBOT",
        "required_columns": {"week_end_sunday", "signups", "self_service", "enterprise", "other_plans"},
    },
    "billing_events": {
        "query_file": "session_billing_events.sql",
        "output_file": "session_billing_events_weekly.csv",
        "env_prefix": "MYSQL_LOGS",
        "required_columns": {"companyId", "week_end_sunday", "conversationType", "event_count", "billable_count"},
    },
    "connect_licenses": {
        "query_file": "connect_licenses_weekly.sql",
        "output_file": "connect_licenses_weekly.csv",
        "env_prefix": "MYSQL_BILLING",
        "required_columns": {"week_end_sunday", "companyId", "connect_licenses"},
    },
    "kyc_steps": {
        "query_file": "kyc_steps_weekly.sql",
        "output_file": "kyc_steps_weekly.csv",
        "env_prefix": "MYSQL_IDENTITY",
        "required_columns": {
            "companyId",
            "week_end_sunday",
            "stepName",
            "status",
            "kyc_steps_count",
        },
    },
}


MONGO_QUERY_JOBS = {
    "workflow_executions": {
        "pipeline_file": "worfklow_executions",
        "output_file": "builder.workflow_executions_logs_weekly.csv",
        "required_columns": {"companyId", "companyName", "week_end_sunday", "execution_count"},
        "env": {
            "uri": ["MONGO_BUILDER_URI", "MONGO_WORKFLOW_URI"],
            "db": ["MONGO_BUILDER_DB", "MONGO_WORKFLOW_DB"],
            "collection": ["MONGO_BUILDER_COLLECTION", "MONGO_WORKFLOW_COLLECTION"],
        },
    },
    "daily_active_users": {
        "pipeline_file": "ai_daily_active_users.json",
        "output_file": "logsM.ai_daily_active_users_weekly.csv",
        "required_columns": {"companyId", "week_end_sunday", "dau_count", "unique_users_count"},
        "env": {
            "uri": ["MONGO_LOGSM_URI", "MONGO_DAU_URI"],
            "db": ["MONGO_LOGSM_DB", "MONGO_DAU_DB"],
            "collection": ["MONGO_LOGSM_COLLECTION", "MONGO_DAU_COLLECTION"],
        },
    },
}


SOURCE_CONFIG = {
    "workflow": {
        "candidates": [
            "builder.workflow_executions_logs_weekly.csv",
            "builder.workflow_executions_logs.csv",
        ],
        "required_columns": {"companyId", "week_end_sunday", "execution_count"},
    },
    "dau": {
        "candidates": [
            "logsM.ai_daily_active_users_weekly.csv",
            "logsM.ai_daily_active_users.csv",
        ],
        "required_columns": {"companyId", "week_end_sunday", "dau_count", "unique_users_count"},
    },
    "billing": {
        "candidates": [
            "session_billing_events_weekly.csv",
            "session_billing_events.csv",
        ],
        "required_columns": {"companyId", "week_end_sunday", "conversationType", "billable_count"},
    },
    "signups": {
        "candidates": ["signups_weekly.csv"],
        "required_columns": {"week_end_sunday", "signups", "self_service", "enterprise", "other_plans"},
    },
    "connect": {
        "candidates": ["connect_licenses_weekly.csv"],
        "required_columns": {"companyId", "week_end_sunday", "connect_licenses"},
    },
    "kyc": {
        "candidates": ["kyc_steps_weekly.csv"],
        "required_columns": {"companyId", "week_end_sunday", "stepName", "status", "kyc_steps_count"},
    },
}


class ValidationError(RuntimeError):
    pass


def log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run query extraction + merge + validation for the weekly dashboard.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to .env credentials file (required unless --skip-extract).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for source and output CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for merged output CSV.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip DB query extraction and only run merge/validation on existing CSVs.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run DB query extraction only (do not merge).",
    )
    parser.add_argument(
        "--strict-date-window",
        action="store_true",
        help="Fail if source files do not share the same min/max week window.",
    )
    parser.add_argument(
        "--run-dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after successful merge.",
    )
    parser.add_argument(
        "--no-git-sync",
        action="store_true",
        help="Disable automatic git commit and push.",
    )
    parser.add_argument(
        "--git-message",
        type=str,
        default="",
        help="Custom git commit message. If omitted, pipeline builds one automatically.",
    )
    parser.add_argument(
        "--git-allow-staged",
        action="store_true",
        help="Allow running git sync even when there are pre-staged changes.",
    )
    parser.add_argument(
        "--allow-billing-fallback",
        action="store_true",
        help=(
            "If billing extraction fails, continue using existing "
            "data/connect_licenses_weekly.csv when available."
        ),
    )
    return parser.parse_args()


def load_module(module_name: str, dependency_label: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ValidationError(
            f"Missing dependency '{dependency_label}'. Install with: pip install -r requirements.txt"
        ) from exc


def get_load_dotenv():
    dotenv_module = load_module("dotenv", "python-dotenv")
    return getattr(dotenv_module, "load_dotenv")


def get_pymysql_module():
    return load_module("pymysql", "pymysql")


def get_mongo_clients():
    pymongo_module = load_module("pymongo", "pymongo")
    bson_json_util = load_module("bson.json_util", "pymongo")
    return getattr(pymongo_module, "MongoClient"), bson_json_util


def require_env_any(keys: Sequence[str], *, default: str | None = None) -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    if default is not None:
        return default
    raise ValidationError(f"Missing environment variable. Set one of: {', '.join(keys)}")


def load_env_file(env_path: Path) -> None:
    load_dotenv = get_load_dotenv()
    if not env_path.exists():
        raise ValidationError(
            f".env file not found at {env_path}. Copy .env.template to .env and fill your credentials."
        )
    load_dotenv(env_path)
    log("ENV", f"Loaded credentials from {env_path}")


def resolve_source_path(data_dir: Path, candidates: Iterable[str]) -> Path:
    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            return path
    expected = ", ".join(str(data_dir / c) for c in candidates)
    raise FileNotFoundError(f"Missing source file. Expected one of: {expected}")


def require_columns(df: pd.DataFrame, required_columns: set[str], source_name: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValidationError(f"[{source_name}] Missing required columns: {missing_sorted}")


def export_mysql_query(job_name: str, job: dict, data_dir: Path) -> None:
    pymysql = get_pymysql_module()
    query_path = QUERIES_DIR / job["query_file"]
    if not query_path.exists():
        raise ValidationError(f"[{job_name}] Query file not found: {query_path}")

    sql = query_path.read_text(encoding="utf-8")
    prefix = job["env_prefix"]
    host = require_env_any([f"{prefix}_HOST"])
    port = int(require_env_any([f"{prefix}_PORT"], default="3306"))
    user = require_env_any([f"{prefix}_USER"])
    password = require_env_any([f"{prefix}_PASSWORD"])
    database = require_env_any([f"{prefix}_DATABASE", f"{prefix}_DB"], default="")

    log(job_name.upper(), f"Running {query_path.name} on {host}:{port} ...")
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database or None,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
        read_timeout=300,
        write_timeout=300,
        charset="utf8mb4",
        autocommit=True,
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description or []]
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=columns if columns else None)
    require_columns(df, job["required_columns"], f"{job_name}_export")
    out_path = data_dir / job["output_file"]
    df.to_csv(out_path, index=False, encoding="utf-8")
    log(job_name.upper(), f"Saved {len(df):,} rows to {out_path.name}")


def load_mongo_pipeline(pipeline_path: Path) -> list[dict]:
    _, bson_json_util = get_mongo_clients()
    raw = pipeline_path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'ISODate\(\s*"([^"]+)"\s*\)',
        r'{"$date":"\1"}',
        raw,
    )
    pipeline = bson_json_util.loads(normalized)
    if not isinstance(pipeline, list):
        raise ValidationError(f"Mongo pipeline must be a list in {pipeline_path}")
    return pipeline


def export_mongo_aggregation(job_name: str, job: dict, data_dir: Path) -> None:
    MongoClient, _ = get_mongo_clients()
    pipeline_path = QUERIES_DIR / job["pipeline_file"]
    if not pipeline_path.exists():
        raise ValidationError(f"[{job_name}] Pipeline file not found: {pipeline_path}")

    env_cfg = job["env"]
    uri = require_env_any(env_cfg["uri"])
    db_name = require_env_any(env_cfg["db"])
    collection_name = require_env_any(env_cfg["collection"])
    pipeline = load_mongo_pipeline(pipeline_path)

    log(job_name.upper(), f"Running {pipeline_path.name} on {db_name}.{collection_name} ...")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    try:
        client.admin.command("ping")
        rows = list(client[db_name][collection_name].aggregate(pipeline, allowDiskUse=True))
    finally:
        client.close()

    df = pd.DataFrame(rows)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    # Preserve expected schema for empty results.
    for col in job["required_columns"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    df = df[list(dict.fromkeys([*df.columns]))]
    require_columns(df, job["required_columns"], f"{job_name}_export")

    out_path = data_dir / job["output_file"]
    df.to_csv(out_path, index=False, encoding="utf-8")
    log(job_name.upper(), f"Saved {len(df):,} rows to {out_path.name}")


def run_query_extraction(data_dir: Path, *, allow_billing_fallback: bool = False) -> None:
    log("EXTRACT", "Starting query extraction")
    for name, job in MYSQL_QUERY_JOBS.items():
        try:
            export_mysql_query(name, job, data_dir)
        except Exception as exc:
            is_billing_job = name == "connect_licenses"
            fallback_file = data_dir / job["output_file"]
            can_fallback = (
                is_billing_job
                and allow_billing_fallback
                and fallback_file.exists()
            )

            if not can_fallback:
                raise

            # Validate fallback file shape before proceeding.
            fallback_df = pd.read_csv(fallback_file)
            require_columns(fallback_df, job["required_columns"], f"{name}_fallback")
            log(
                "EXTRACT",
                (
                    f"WARNING: {name} extraction failed ({exc}). "
                    f"Using existing {fallback_file.name}."
                ),
            )
    for name, job in MONGO_QUERY_JOBS.items():
        export_mongo_aggregation(name, job, data_dir)
    log("EXTRACT", "All source queries completed")


def normalize_company_id(series: pd.Series) -> pd.Series:
    series = series.astype("string").str.strip()
    series = series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.notna()
    normalized = series.copy()
    normalized.loc[mask] = numeric.loc[mask].astype("Int64").astype("string")
    return normalized


def normalize_week_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    out["week_end_sunday"] = pd.to_datetime(out["week_end_sunday"], errors="coerce").dt.normalize()
    invalid_rows = out["week_end_sunday"].isna().sum()
    if invalid_rows:
        raise ValidationError(f"[{source_name}] Found {invalid_rows} rows with invalid week_end_sunday values.")
    return out


def to_numeric_columns(df: pd.DataFrame, columns: Iterable[str], source_name: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if out[col].isna().any():
                bad_count = out[col].isna().sum()
                raise ValidationError(f"[{source_name}] Column '{col}' has {bad_count} non-numeric values.")
    return out


def check_duplicate_grain(df: pd.DataFrame, keys: list[str], source_name: str) -> None:
    duplicate_count = df.duplicated(subset=keys).sum()
    if duplicate_count:
        keys_str = " + ".join(keys)
        raise ValidationError(f"[{source_name}] Found {duplicate_count} duplicate rows for grain {keys_str}.")


def summarize_date_windows(datasets: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, df in datasets.items():
        if df.empty:
            continue
        windows[name] = (df["week_end_sunday"].min(), df["week_end_sunday"].max())
    return windows


def slug_plan_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "unknown_plan"


def prepare_workflow(df_workflow: pd.DataFrame) -> pd.DataFrame:
    df = df_workflow.copy()
    if "companyName" not in df.columns:
        df["companyName"] = "Unknown"
    df["companyName"] = df["companyName"].fillna("Unknown")
    df["companyId"] = normalize_company_id(df["companyId"])
    df = normalize_week_column(df, "workflow")
    df = to_numeric_columns(df, ["execution_count"], "workflow")

    return (
        df.groupby(["companyId", "week_end_sunday"], as_index=False)
        .agg({"companyName": "first", "execution_count": "sum"})
        .sort_values(["week_end_sunday", "companyId"])
    )


def prepare_dau(df_dau: pd.DataFrame) -> pd.DataFrame:
    df = df_dau.copy()
    df["companyId"] = normalize_company_id(df["companyId"])
    df = normalize_week_column(df, "dau")
    df = to_numeric_columns(df, ["dau_count", "unique_users_count"], "dau")

    aggregated = (
        df.groupby(["companyId", "week_end_sunday"], as_index=False)[["dau_count", "unique_users_count"]]
        .sum()
        .sort_values(["week_end_sunday", "companyId"])
    )
    check_duplicate_grain(aggregated, ["companyId", "week_end_sunday"], "dau")
    return aggregated


def prepare_billing(df_billing: pd.DataFrame) -> pd.DataFrame:
    df = df_billing.copy()
    df["companyId"] = normalize_company_id(df["companyId"])
    df = normalize_week_column(df, "billing")
    df = to_numeric_columns(df, ["billable_count"], "billing")

    df["conversationType"] = df["conversationType"].astype("string").str.strip().str.upper()
    df["conversationType"] = df["conversationType"].replace({"MARKETING_LITE": "MARKETING"})
    valid_types = {"MARKETING", "UTILITY", "AUTHENTICATION"}
    df = df[df["conversationType"].isin(valid_types)].copy()

    grouped = (
        df.groupby(["companyId", "week_end_sunday", "conversationType"], as_index=False)["billable_count"]
        .sum()
        .sort_values(["week_end_sunday", "companyId", "conversationType"])
    )
    check_duplicate_grain(grouped, ["companyId", "week_end_sunday", "conversationType"], "billing")

    pivot = (
        grouped.pivot_table(
            index=["companyId", "week_end_sunday"],
            columns="conversationType",
            values="billable_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in ["MARKETING", "UTILITY", "AUTHENTICATION"]:
        if col not in pivot.columns:
            pivot[col] = 0

    return pivot[["companyId", "week_end_sunday", "MARKETING", "UTILITY", "AUTHENTICATION"]]


def build_plan_list(row: pd.Series, rename_map: dict[str, str]) -> str:
    active = []
    for original_name, slug_col in rename_map.items():
        if row.get(slug_col, 0) > 0:
            active.append(str(original_name))
    return ", ".join(sorted(active))


def prepare_connect(df_connect: pd.DataFrame) -> pd.DataFrame:
    df = df_connect.copy()
    df["companyId"] = normalize_company_id(df["companyId"])
    df = normalize_week_column(df, "connect")
    df = to_numeric_columns(df, ["connect_licenses"], "connect")

    if "product_name" not in df.columns:
        return (
            df.groupby(["companyId", "week_end_sunday"], as_index=False)["connect_licenses"]
            .sum()
            .sort_values(["week_end_sunday", "companyId"])
        )

    df["product_name"] = df["product_name"].fillna("Unknown Plan").astype(str).str.strip()
    grouped = (
        df.groupby(["companyId", "week_end_sunday", "product_name"], as_index=False)["connect_licenses"]
        .sum()
        .sort_values(["week_end_sunday", "companyId", "product_name"])
    )

    connect_pivot = grouped.pivot_table(
        index=["companyId", "week_end_sunday"],
        columns="product_name",
        values="connect_licenses",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    plan_cols_original = [c for c in connect_pivot.columns if c not in ["companyId", "week_end_sunday"]]
    rename_map = {c: f"connect_plan_{slug_plan_name(c)}" for c in plan_cols_original}
    connect_pivot = connect_pivot.rename(columns=rename_map)

    plan_cols_slug = list(rename_map.values())
    if plan_cols_slug:
        connect_pivot["connect_licenses"] = connect_pivot[plan_cols_slug].sum(axis=1)
        connect_pivot["connect_plan_list"] = connect_pivot.apply(build_plan_list, axis=1, rename_map=rename_map)
    else:
        connect_pivot["connect_licenses"] = 0
        connect_pivot["connect_plan_list"] = ""

    return connect_pivot.sort_values(["week_end_sunday", "companyId"])


def prepare_kyc(df_kyc: pd.DataFrame) -> pd.DataFrame:
    df = df_kyc.copy()
    df["companyId"] = normalize_company_id(df["companyId"])
    df = normalize_week_column(df, "kyc")
    df = to_numeric_columns(df, ["kyc_steps_count"], "kyc")

    df["stepName"] = df["stepName"].astype("string").str.strip().str.lower()
    df["status"] = df["status"].astype("string").str.strip().str.lower()
    df["stepName"] = df["stepName"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    df["status"] = df["status"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    df = df[df["stepName"].notna() & df["status"].notna()].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "companyId",
                "week_end_sunday",
                "kyc_document_check_total",
                "kyc_liveness_total",
                "kyc_facematch_total",
                "kyc_total",
            ]
        )

    grouped = (
        df.groupby(["companyId", "week_end_sunday", "stepName", "status"], as_index=False)["kyc_steps_count"]
        .sum()
        .sort_values(["week_end_sunday", "companyId", "stepName", "status"])
    )
    check_duplicate_grain(grouped, ["companyId", "week_end_sunday", "stepName", "status"], "kyc")

    status_grouped = grouped.copy()
    status_grouped["kyc_col"] = (
        "kyc_"
        + status_grouped["stepName"].map(slug_plan_name)
        + "_"
        + status_grouped["status"].map(slug_plan_name)
    )
    status_pivot = (
        status_grouped.pivot_table(
            index=["companyId", "week_end_sunday"],
            columns="kyc_col",
            values="kyc_steps_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    stage_grouped = (
        grouped.groupby(["companyId", "week_end_sunday", "stepName"], as_index=False)["kyc_steps_count"]
        .sum()
    )
    stage_grouped["stage_col"] = "kyc_" + stage_grouped["stepName"].map(slug_plan_name) + "_total"
    stage_pivot = (
        stage_grouped.pivot_table(
            index=["companyId", "week_end_sunday"],
            columns="stage_col",
            values="kyc_steps_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    totals = (
        grouped.groupby(["companyId", "week_end_sunday"], as_index=False)["kyc_steps_count"]
        .sum()
        .rename(columns={"kyc_steps_count": "kyc_total"})
    )

    kyc = status_pivot.merge(stage_pivot, on=["companyId", "week_end_sunday"], how="outer")
    kyc = kyc.merge(totals, on=["companyId", "week_end_sunday"], how="left")

    for col in ["kyc_document_check_total", "kyc_liveness_total", "kyc_facematch_total", "kyc_total"]:
        if col not in kyc.columns:
            kyc[col] = 0

    kyc = kyc.sort_values(["week_end_sunday", "companyId"])
    check_duplicate_grain(kyc, ["companyId", "week_end_sunday"], "kyc_prepared")
    return kyc


def prepare_signups(df_signups: pd.DataFrame) -> pd.DataFrame:
    df = df_signups.copy()
    df = normalize_week_column(df, "signups")
    numeric_cols = ["signups", "self_service", "enterprise", "other_plans"]
    df = to_numeric_columns(df, numeric_cols, "signups")

    aggregated = (
        df.groupby("week_end_sunday", as_index=False)[numeric_cols]
        .sum()
        .sort_values("week_end_sunday")
    )
    check_duplicate_grain(aggregated, ["week_end_sunday"], "signups")
    return aggregated


def merge_sources(
    workflow: pd.DataFrame,
    billing: pd.DataFrame,
    dau: pd.DataFrame,
    connect: pd.DataFrame,
    kyc: pd.DataFrame,
    signups: pd.DataFrame,
) -> pd.DataFrame:
    company_keys = ["companyId", "week_end_sunday"]
    merged = workflow.merge(billing, on=company_keys, how="outer")
    merged = merged.merge(dau, on=company_keys, how="outer")
    merged = merged.merge(connect, on=company_keys, how="outer")
    merged = merged.merge(kyc, on=company_keys, how="outer")
    merged = merged.merge(signups, on="week_end_sunday", how="left")

    # Keep only complete, closed weekly periods (last completed Sunday cutoff).
    today = pd.Timestamp.today().normalize()
    last_completed_week_end = today - pd.Timedelta(days=today.weekday() + 1)
    merged = merged[merged["week_end_sunday"].notna()]
    merged = merged[merged["week_end_sunday"] <= last_completed_week_end].copy()

    plan_cols = [c for c in merged.columns if c.startswith("connect_plan_") and c != "connect_plan_list"]
    kyc_cols = [c for c in merged.columns if c.startswith("kyc_")]
    numeric_cols = [
        "execution_count",
        "dau_count",
        "unique_users_count",
        "MARKETING",
        "UTILITY",
        "AUTHENTICATION",
        "connect_licenses",
        "signups",
        "self_service",
        "enterprise",
        "other_plans",
    ] + plan_cols + kyc_cols

    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    if "companyName" in merged.columns:
        merged["companyName"] = merged["companyName"].fillna("Unknown")
    else:
        merged["companyName"] = "Unknown"

    if "connect_plan_list" in merged.columns:
        merged["connect_plan_list"] = merged["connect_plan_list"].fillna("")

    merged = merged.sort_values(["week_end_sunday", "companyId"])
    merged["week_end_sunday"] = merged["week_end_sunday"].dt.strftime("%Y-%m-%d")

    check_duplicate_grain(merged, ["companyId", "week_end_sunday"], "merged_output")
    return merged


def run_merge_and_validate(data_dir: Path, output_path: Path, strict_date_window: bool) -> None:
    log("MERGE", f"Reading source CSV files from: {data_dir}")
    raw_sources: dict[str, pd.DataFrame] = {}

    for source_name, cfg in SOURCE_CONFIG.items():
        source_path = resolve_source_path(data_dir, cfg["candidates"])
        df = pd.read_csv(source_path)
        require_columns(df, cfg["required_columns"], source_name)
        raw_sources[source_name] = df
        log("MERGE", f"- {source_name}: {source_path.name} ({len(df):,} rows)")

    workflow = prepare_workflow(raw_sources["workflow"])
    billing = prepare_billing(raw_sources["billing"])
    dau = prepare_dau(raw_sources["dau"])
    connect = prepare_connect(raw_sources["connect"])
    kyc = prepare_kyc(raw_sources["kyc"])
    signups = prepare_signups(raw_sources["signups"])

    prepared_sources = {
        "workflow": workflow,
        "billing": billing,
        "dau": dau,
        "connect": connect,
        "kyc": kyc,
        "signups": signups,
    }

    windows = summarize_date_windows(prepared_sources)
    if windows:
        print("\nWeekly coverage:")
        for name, (start, end) in windows.items():
            print(f"- {name}: {start.date()} -> {end.date()}")

        starts = {window[0] for window in windows.values()}
        ends = {window[1] for window in windows.values()}
        if len(starts) > 1 or len(ends) > 1:
            msg = "Source files do not share the same min/max week window."
            if strict_date_window:
                raise ValidationError(msg)
            print(f"WARNING: {msg}")

    merged = merge_sources(workflow, billing, dau, connect, kyc, signups)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\nMerged output:")
    print(f"- rows: {len(merged):,}")
    print(f"- columns: {len(merged.columns)}")
    print(f"- unique companies: {merged['companyId'].nunique():,}")
    print(f"- unique weeks: {merged['week_end_sunday'].nunique():,}")
    print(f"- saved to: {output_path}")


def run_git_command(args: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def get_repo_root() -> Path | None:
    proc = run_git_command(["git", "rev-parse", "--show-toplevel"], PROJECT_ROOT)
    if proc.returncode != 0:
        return None
    return Path(proc.stdout.strip())


def list_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def collect_generated_files(data_dir: Path, output_path: Path) -> list[Path]:
    files: list[Path] = []
    for job in MYSQL_QUERY_JOBS.values():
        files.append(data_dir / job["output_file"])
    for job in MONGO_QUERY_JOBS.values():
        files.append(data_dir / job["output_file"])
    files.append(output_path)

    unique_existing: list[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            unique_existing.append(resolved)
            seen.add(resolved)
    return unique_existing


def build_default_commit_message(output_path: Path) -> str:
    if output_path.exists():
        try:
            merged = pd.read_csv(output_path, usecols=["week_end_sunday"])
            if not merged.empty:
                latest_week = str(merged["week_end_sunday"].max())
                return f"chore: refresh weekly dashboard data ({latest_week})"
        except Exception:
            pass
    return "chore: refresh weekly dashboard data"


def run_git_sync(
    *,
    data_dir: Path,
    output_path: Path,
    custom_message: str,
    allow_staged: bool,
) -> None:
    repo_root = get_repo_root()
    if repo_root is None:
        raise ValidationError("Git sync requested but this folder is not inside a git repository.")

    pre_staged = run_git_command(["git", "diff", "--cached", "--name-only"], repo_root)
    if pre_staged.returncode != 0:
        raise ValidationError(f"Unable to inspect staged changes: {pre_staged.stderr.strip()}")
    staged_before = list_nonempty_lines(pre_staged.stdout)
    if staged_before and not allow_staged:
        raise ValidationError(
            "Git sync blocked: there are already staged changes. "
            "Commit/unstage them first, or run with --git-allow-staged."
        )

    generated_files = collect_generated_files(data_dir, output_path)
    if not generated_files:
        log("GIT", "No generated files found to sync.")
        return

    rel_paths: list[str] = []
    for file_path in generated_files:
        try:
            rel_paths.append(str(file_path.relative_to(repo_root)))
        except ValueError:
            continue

    if not rel_paths:
        log("GIT", "No generated files are inside this git repository.")
        return

    add_proc = run_git_command(["git", "add", "--", *rel_paths], repo_root)
    if add_proc.returncode != 0:
        raise ValidationError(f"git add failed: {add_proc.stderr.strip()}")

    staged_generated = run_git_command(["git", "diff", "--cached", "--name-only", "--", *rel_paths], repo_root)
    if staged_generated.returncode != 0:
        raise ValidationError(f"Unable to inspect staged generated files: {staged_generated.stderr.strip()}")
    generated_changed = list_nonempty_lines(staged_generated.stdout)
    if not generated_changed:
        log("GIT", "No generated file changes to commit.")
        return

    message = custom_message.strip() or build_default_commit_message(output_path)
    commit_proc = run_git_command(["git", "commit", "-m", message], repo_root)
    if commit_proc.returncode != 0:
        raise ValidationError(f"git commit failed: {commit_proc.stderr.strip() or commit_proc.stdout.strip()}")
    log("GIT", f"Committed {len(generated_changed)} file(s): {message}")

    push_proc = run_git_command(["git", "push"], repo_root)
    if push_proc.returncode != 0:
        raise ValidationError(f"git push failed: {push_proc.stderr.strip() or push_proc.stdout.strip()}")
    log("GIT", "Pushed changes to remote.")


def run() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  WEEKLY DASHBOARD PIPELINE - STARTING")
    print("=" * 60)

    if args.extract_only and args.run_dashboard:
        raise ValidationError("--extract-only and --run-dashboard cannot be used together.")

    if not args.skip_extract:
        load_env_file(args.env_file.resolve())
        run_query_extraction(data_dir, allow_billing_fallback=args.allow_billing_fallback)
    else:
        log("EXTRACT", "Skipped by --skip-extract")

    if args.extract_only:
        if not args.no_git_sync:
            run_git_sync(
                data_dir=data_dir,
                output_path=output_path,
                custom_message=args.git_message,
                allow_staged=args.git_allow_staged,
            )
        log("DONE", "Extraction completed (--extract-only).")
        return

    run_merge_and_validate(data_dir, output_path, args.strict_date_window)

    if not args.no_git_sync:
        run_git_sync(
            data_dir=data_dir,
            output_path=output_path,
            custom_message=args.git_message,
            allow_staged=args.git_allow_staged,
        )

    if args.run_dashboard:
        print("\nLaunching Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py")], check=False)


if __name__ == "__main__":
    try:
        run()
    except (ValidationError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
