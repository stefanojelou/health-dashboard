-- Marketplace app installation events
-- Run in MySQL (database: marketplace)
-- Export to CSV: data/marketplace_installations_events.csv
-- Expected columns:
-- installation_id, userId, userType, installedByUserId, mapping_user_id,
-- appId, app_name, installed_at, installed_date, week_end_sunday, updatedAt

SELECT
    i.id AS installation_id,
    i.userId,
    i.userType,
    i.installedByUserId,
    CASE
        WHEN UPPER(i.userType) = 'ORGANIZATION' THEN i.installedByUserId
        ELSE i.userId
    END AS mapping_user_id,
    i.appId,
    a.name AS app_name,
    i.createdAt AS installed_at,
    DATE(i.createdAt) AS installed_date,
    DATE_ADD(DATE(i.createdAt), INTERVAL (6 - WEEKDAY(i.createdAt)) DAY) AS week_end_sunday,
    i.updatedAt
FROM marketplace.user_app_installations i
INNER JOIN marketplace.apps a
    ON a.id = i.appId
WHERE i.deletedAt IS NULL
  AND i.createdAt >= '2025-01-01'
ORDER BY i.createdAt ASC, i.id ASC;
