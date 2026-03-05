-- Weekly active Connect licenses by company
-- Run in MySQL (database: billing)
-- Export to CSV: data/connect_licenses_weekly.csv
-- Expected columns: week_end_sunday, companyId, connect_licenses
-- Note: uses subscription item quantity as license count.

WITH RECURSIVE week_calendar AS (
    SELECT DATE_ADD(DATE('2025-01-01'), INTERVAL (6 - WEEKDAY('2025-01-01')) DAY) AS week_end_sunday
    UNION ALL
    SELECT DATE_ADD(week_end_sunday, INTERVAL 7 DAY)
    FROM week_calendar
    WHERE week_end_sunday < DATE_ADD(CURDATE(), INTERVAL 14 DAY)
),
connect_items AS (
    SELECT
        s.id AS subscription_id,
        s.company_id AS companyId,
        DATE(s.current_period_start) AS period_start,
        DATE(COALESCE(s.canceled_at, s.cancel_at, s.current_period_end, NOW())) AS period_end,
        COALESCE(NULLIF(si.quantity, 0), 1) AS license_qty
    FROM billing.subscriptions s
    LEFT JOIN billing.subscription_items si
        ON si.subscription_id = s.id
    LEFT JOIN billing.stripe_products sp
        ON sp.id = si.product_id
    WHERE s.company_id IS NOT NULL
      AND sp.name LIKE '%Connect%'
      AND UPPER(s.status) IN ('ACTIVE', 'TRIALING')
)
SELECT
    wc.week_end_sunday,
    ci.companyId,
    SUM(ci.license_qty) AS connect_licenses
FROM week_calendar wc
JOIN connect_items ci
    ON wc.week_end_sunday BETWEEN ci.period_start AND ci.period_end
GROUP BY wc.week_end_sunday, ci.companyId
ORDER BY wc.week_end_sunday ASC, ci.companyId ASC;
