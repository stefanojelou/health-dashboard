# Dashboard Update Guide (Weekly)

This guide explains the weekly refresh process for `salud_cuentas/app.py`.

## 1) Weekly metrics in scope

- Workflow Executions
- Daily Active Users
- Licencias de Connect
- WhatsApp messages: `MARKETING`, `UTILITY`, `AUTHENTICATION`
- Sign Ups
- Nuevos SMBs (`other_plans` from signups split)

Deferred for now: KYC, Firmas, Validacion gubernamental, GMV by gateway/company.

## 2) Query files to run

Run all query files under `salud_cuentas/queries/` and export to `salud_cuentas/data/` with exact names:

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

## 3) Date and grain standards

- Use weekly grain with `week_end_sunday` (format `YYYY-MM-DD`).
- Keep the same start date across all queries.
- Keep `companyId` data type consistent across all exports.
- Use UTF-8 CSV and overwrite old files.

## 4) Notebook merge step

1. Open `salud_cuentas/salud_cuentas.ipynb`.
2. Run all cells.
3. Confirm export file exists: `data/merged_weekly_data.csv`.

The notebook merges company-week datasets and joins weekly signups split (`signups`, `self_service`, `enterprise`, `other_plans`).

## 5) Dashboard run

Install dependencies (first time):

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## 6) Weekly validation checklist

Before opening the dashboard:

1. Confirm all 5 weekly CSV files exist in `data/`.
2. Confirm required columns are present exactly as listed above.
3. Confirm no duplicate rows at these grains:
   - workflow: `companyId + week_end_sunday`
   - dau: `companyId + week_end_sunday`
   - connect: `companyId + week_end_sunday`
   - billing: `companyId + week_end_sunday + conversationType`
   - signups: `week_end_sunday`
4. Confirm numeric fields are numeric:
   - `execution_count`, `dau_count`, `unique_users_count`, `billable_count`, `connect_licenses`
   - `signups`, `self_service`, `enterprise`, `other_plans`
5. Confirm all weekly files cover the same date window.

## 7) Dashboard smoke checks

In `app.py`, verify:

- **Overview**: cards render for Workflow, DAU, Connect, WhatsApp types, Sign Ups, Nuevos SMBs.
- **Company Lookup**: metrics and WoW deltas render for at least one company.
- **Distributions**: all metric options load and top company charts are populated.
- **WoW Analysis**: weekly trend chart, WoW chart, gainers/decliners tables render.

If values look too low/high, re-check:

- export date ranges in query files
- workflow aggregation logic (`$sum: $total` vs `$sum: 1`)
- billing conversation type filter including `AUTHENTICATION`.

