-- Weekly signup counts from chatbot.companies
-- Run in MySQL (database: chatbot)
-- Export to CSV: data/signups_weekly.csv
-- Expected columns: week_end_sunday, signups, self_service, enterprise, other_plans

SELECT
    DATE_ADD(DATE(c.createdAt), INTERVAL (6 - WEEKDAY(c.createdAt)) DAY) AS week_end_sunday,
    COUNT(*) AS signups,
    SUM(CASE WHEN c.plan = 'SELF_SERVICE' THEN 1 ELSE 0 END) AS self_service,
    SUM(CASE WHEN c.plan = 'ENTERPRISE' THEN 1 ELSE 0 END) AS enterprise,
    SUM(CASE WHEN c.plan NOT IN ('SELF_SERVICE', 'ENTERPRISE') THEN 1 ELSE 0 END) AS other_plans
FROM chatbot.companies c
WHERE c.createdAt >= '2025-01-01'
  -- Exclude internal/test companies by company fields
  AND (
    c.name IS NULL OR c.name = ''
    OR (
      LOWER(c.name) NOT LIKE '%jelou%'
      AND LOWER(c.name) NOT LIKE '%impersonate%'
      AND LOWER(c.name) NOT LIKE '%test%'
      AND LOWER(c.name) NOT LIKE '%prueba%'
    )
  )
  AND (
    c.slug IS NULL OR c.slug = ''
    OR (
      LOWER(c.slug) NOT LIKE '%jelou%'
      AND LOWER(c.slug) NOT LIKE '%impersonate%'
      AND LOWER(c.slug) NOT LIKE '%test%'
      AND LOWER(c.slug) NOT LIKE '%prueba%'
    )
  )
  AND (
    c.email IS NULL OR c.email = ''
    OR (
      LOWER(c.email) NOT LIKE '%jelou%'
      AND LOWER(c.email) NOT LIKE '%impersonate%'
      AND LOWER(c.email) NOT LIKE '%test%'
      AND LOWER(c.email) NOT LIKE '%prueba%'
    )
  )
  -- Exclude companies whose ROOT user has internal/test email
  AND NOT EXISTS (
    SELECT 1
    FROM chatbot.users u
    WHERE u.companyId = c.id
      AND u.deletedAt IS NULL
      AND u.isRoot = 1
      AND u.email IS NOT NULL
      AND TRIM(u.email) <> ''
      AND (
        LOWER(u.email) LIKE '%jelou%'
        OR LOWER(u.email) LIKE '%impersonate%'
        OR LOWER(u.email) LIKE '%test%'
        OR LOWER(u.email) LIKE '%prueba%'
      )
  )
GROUP BY week_end_sunday
ORDER BY week_end_sunday ASC;
