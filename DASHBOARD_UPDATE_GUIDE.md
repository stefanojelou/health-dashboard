# Dashboard Update Guide (Weekly)

This guide explains the weekly refresh process for `salud_cuentas/app.py`.

## 1) Weekly metrics in scope

- Workflow Executions
- Daily Active Users
- Licencias de Connect
- WhatsApp messages: `MARKETING`, `UTILITY`, `AUTHENTICATION`
- Sign Ups
- Nuevos SMBs (`other_plans` from signups split)

Deferred for now: Firmas, Validacion gubernamental, GMV by gateway/company.

## 2) Query files used by the pipeline

`update_dashboard_data.py` executes all query files under `salud_cuentas/queries/` and exports to `salud_cuentas/data/`.

1. `queries/worfklow_executions`  
   Export: `data/builder.workflow_executions_logs_weekly.csv`  
   Columns: `companyId`, `companyName`, `week_end_sunday`, `execution_count`

2. `queries/ai_daily_active_users.json`  
   Export: `data/logsM.ai_daily_active_users_weekly.csv`  
   Columns: `companyId`, `week_end_sunday`, `dau_count`, `unique_users_count`

3. `queries/session_billing_events.sql`  
   Export: `data/session_billing_events_weekly.csv`  
   Columns: `companyId`, `week_end_sunday`, `conversationType`, `event_count`, `billable_count`  
   Includes: `MARKETING`, `MARKETING_LITE`, `UTILITY`, `AUTHENTICATION`

4. `queries/signups_weekly.sql`  
   Export: `data/signups_weekly.csv`  
   Columns: `week_end_sunday`, `signups`, `self_service`, `enterprise`, `other_plans`

5. `queries/connect_licenses_weekly.sql`  
   Export: `data/connect_licenses_weekly.csv`  
   Columns: `week_end_sunday`, `companyId`, `connect_licenses`

6. `queries/kyc_steps_weekly.sql`  
   Export: `data/kyc_steps_weekly.csv`  
   Columns: `companyId`, `companyName`, `week_end_sunday`, `stepName`, `status`, `kyc_steps_count`
   Note: runs only on `identity` (no cross-schema join to `chatbot` required).

## 3) Date and grain standards

- Use weekly grain with `week_end_sunday` (format `YYYY-MM-DD`).
- Keep the same start date across all queries.
- Keep `companyId` data type consistent across all exports.
- Use UTF-8 CSV and overwrite old files.

## 4) One-command automated pipeline (recommended)

Prepare credentials:

1. Copy `.env.template` to `.env`.
2. Fill all DB credentials.
3. Include Identity DB credentials for KYC extraction:
   - `MYSQL_IDENTITY_HOST`
   - `MYSQL_IDENTITY_PORT`
   - `MYSQL_IDENTITY_USER`
   - `MYSQL_IDENTITY_PASSWORD`
   - `MYSQL_IDENTITY_DATABASE`

Install dependencies (first time or after updates):

```bash
pip install -r requirements.txt
```

Run the updater:

```bash
python update_dashboard_data.py
```

What it does:
- loads credentials from `.env`
- runs 4 MySQL queries + 2 Mongo pipelines from `queries/`
- exports all weekly CSV sources into `data/`
- validates required columns, numeric fields, weekly dates, and duplicate grains
- rebuilds the canonical merged file for Streamlit
- saves `data/merged_weekly_data.csv`
- commits and pushes generated CSV updates to your current git branch

Useful flags:

```bash
# Skip query extraction and only merge existing CSVs
python update_dashboard_data.py --skip-extract

# Run extraction only (no merge)
python update_dashboard_data.py --extract-only

# Disable automatic git commit/push for this run
python update_dashboard_data.py --no-git-sync

# Use a custom commit message
python update_dashboard_data.py --git-message "chore: refresh weekly dashboard data"

# If billing DB is unreachable, keep existing billing CSV
python update_dashboard_data.py --allow-billing-fallback
```

Optional strict coverage validation:

```bash
python update_dashboard_data.py --strict-date-window
```

Optional one-command refresh + dashboard launch:

```bash
python update_dashboard_data.py --run-dashboard
```

## 5) Notebook merge step (optional / legacy)

If you want to keep the notebook flow:

1. Open `salud_cuentas/salud_cuentas.ipynb`.
2. Run all cells.
3. Confirm export file exists: `data/merged_weekly_data.csv`.

The notebook and the script should produce the same canonical file.

## 6) Dashboard run

Run:

```bash
streamlit run app.py
```

## 7) Weekly validation checklist

Before opening the dashboard (automatically covered by `update_dashboard_data.py`):

1. Confirm all 6 weekly CSV files exist in `data/`.
2. Confirm required columns are present exactly as listed above.
3. Confirm no duplicate rows at these grains:
   - workflow: `companyId + week_end_sunday`
   - dau: `companyId + week_end_sunday`
   - connect: `companyId + week_end_sunday`
   - billing: `companyId + week_end_sunday + conversationType`
   - kyc: `companyId + week_end_sunday + stepName + status`
   - signups: `week_end_sunday`
4. Confirm numeric fields are numeric:
   - `execution_count`, `dau_count`, `unique_users_count`, `billable_count`, `connect_licenses`
   - `kyc_steps_count` (in source export) and generated `kyc_*` columns in merged output
   - `signups`, `self_service`, `enterprise`, `other_plans`
5. Confirm all weekly files cover the same date window.

## 8) Dashboard smoke checks

In `app.py`, verify:

- **Overview**: cards render for Workflow, DAU, Connect, WhatsApp types, Sign Ups, Nuevos SMBs.
- **Overview**: KYC stacked chart by stage renders and respects Weekly/Monthly selector.
- **Company Lookup**: metrics and WoW deltas render for at least one company.
- **Company Lookup**: KYC stage cards, stacked trend, and status detail table render for at least one company with KYC.
- **Distributions**: all metric options load and top company charts are populated.
- **WoW Analysis**: weekly trend chart, WoW chart, gainers/decliners tables render.

If values look too low/high, re-check:

- export date ranges in query files
- workflow aggregation logic (`$sum: $total` vs `$sum: 1`)
- billing conversation type filter including `AUTHENTICATION`.

