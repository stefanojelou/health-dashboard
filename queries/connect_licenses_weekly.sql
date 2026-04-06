-- Weekly active Connect licenses by company and plan tier
-- Run in MySQL (database: billing)
-- Export to CSV: data/connect_licenses_weekly.csv
-- Expected columns:
--   week_end_sunday, companyId, plan_tier_name, plan_tier_display_name, connect_licenses
-- Note: uses subscription item quantity as license count.
-- Connect platform_product_id provided by billing team.

WITH RECURSIVE week_calendar AS (
    SELECT DATE_ADD(DATE('2025-01-01'), INTERVAL (6 - WEEKDAY('2025-01-01')) DAY) AS week_end_sunday
    UNION ALL
    SELECT DATE_ADD(week_end_sunday, INTERVAL 7 DAY)
    FROM week_calendar
    WHERE week_end_sunday < DATE_ADD(CURDATE(), INTERVAL 14 DAY)
),
connect_items AS (
    SELECT
        s.company_id AS companyId,
        pt.name AS plan_tier_name,
        COALESCE(NULLIF(pt.display_name, ''), pt.name, 'Unknown Plan') AS plan_tier_display_name,
        DATE(s.current_period_start) AS period_start,
        DATE(COALESCE(s.canceled_at, s.cancel_at, s.current_period_end, NOW())) AS period_end,
        COALESCE(NULLIF(si.quantity, 0), 1) AS license_qty
    FROM billing.subscriptions s
    INNER JOIN billing.subscription_items si
        ON si.subscription_id = s.id
    INNER JOIN billing.plan_tiers pt
        ON pt.id = si.plan_tier_id
    WHERE s.company_id IS NOT NULL
      AND si.platform_product_id = '01KFNGRVQ1PAN0FGTMCZ959P9C'
      AND UPPER(s.status) = 'ACTIVE'
)
SELECT
    wc.week_end_sunday,
    ci.companyId,
    ci.plan_tier_name,
    ci.plan_tier_display_name,
    SUM(ci.license_qty) AS connect_licenses
FROM week_calendar wc
JOIN connect_items ci
    ON wc.week_end_sunday BETWEEN ci.period_start AND ci.period_end
GROUP BY
    wc.week_end_sunday,
    ci.companyId,
    ci.plan_tier_name,
    ci.plan_tier_display_name
ORDER BY
    wc.week_end_sunday ASC,
    ci.companyId ASC,
    ci.plan_tier_display_name ASC,
    ci.plan_tier_name ASC;
