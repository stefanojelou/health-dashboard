SELECT 
    companyId,
    DATE_FORMAT(createdAt, '%Y-%m') as month,
    conversationType,
    COUNT(*) as event_count,
    SUM(isBillable) as billable_count
FROM logs.session_billing_events
WHERE conversationType IN ('MARKETING', 'MARKETING_LITE', 'UTILITY')
  AND createdAt >= '2025-06-26' 
  AND createdAt < '2026-01-01'
GROUP BY companyId, month, conversationType
ORDER BY companyId, month;