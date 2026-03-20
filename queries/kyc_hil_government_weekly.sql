-- Weekly KYC government checks and HIL metrics by company
-- Run in MySQL (database: identity)
-- Export to CSV: data/kyc_hil_government_weekly.csv
-- Expected columns:
--   companyId, companyName, week_end_sunday,
--   kyc_sessions_total, kyc_gov_entity_queries,
--   kyc_hil_sessions, kyc_hil_approved, kyc_hil_disapproved

SELECT
    b.companyId,
    CONCAT('Company ', b.companyId) AS companyName,
    DATE_ADD(DATE(bs.createdAt), INTERVAL (6 - WEEKDAY(bs.createdAt)) DAY) AS week_end_sunday,
    COUNT(DISTINCT b.id) AS kyc_sessions_total,
    SUM(
        CASE
            WHEN bs.stepName = 'document_check' AND dcs.entityResponse IS NOT NULL THEN 1
            ELSE 0
        END
    ) AS kyc_gov_entity_queries,
    COUNT(DISTINCT CASE WHEN bs.enableHumanInLoop = 1 THEN b.id END) AS kyc_hil_sessions,
    SUM(
        CASE
            WHEN bs.enableHumanInLoop = 1 AND bs.status = 'Approved' THEN 1
            ELSE 0
        END
    ) AS kyc_hil_approved,
    SUM(
        CASE
            WHEN bs.enableHumanInLoop = 1 AND bs.status = 'Disapproved' THEN 1
            ELSE 0
        END
    ) AS kyc_hil_disapproved
FROM identity.biometric_steps bs
JOIN identity.biometrics b
    ON b.id = bs.biometricId
LEFT JOIN identity.document_check_steps dcs
    ON dcs.id = bs.documentCheckId
WHERE bs.createdAt >= '2025-01-01'
  AND b.companyId IS NOT NULL
  AND bs.stepName IN ('document_check', 'liveness', 'facematch')
GROUP BY
    b.companyId,
    companyName,
    week_end_sunday
ORDER BY
    b.companyId,
    week_end_sunday;
