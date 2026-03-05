-- Weekly WhatsApp billing events by conversation type
-- Run in MySQL (database: logs)
-- Export to CSV: data/session_billing_events_weekly.csv
-- Expected columns: companyId, week_end_sunday, conversationType, event_count, billable_count

SELECT
    companyId,
    DATE_ADD(DATE(createdAt), INTERVAL (6 - WEEKDAY(createdAt)) DAY) AS week_end_sunday,
    conversationType,
    COUNT(*) AS event_count,
    SUM(isBillable) AS billable_count
FROM logs.session_billing_events
WHERE conversationType IN ('MARKETING', 'MARKETING_LITE', 'UTILITY', 'AUTHENTICATION')
  AND companyId IS NOT NULL
  AND createdAt >= '2025-01-01'
GROUP BY companyId, week_end_sunday, conversationType
ORDER BY companyId, week_end_sunday, conversationType;