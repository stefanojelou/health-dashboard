-- Weekly KYC government checks and HIL metrics by company
-- Run in MySQL (database: identity)
-- Export to CSV: data/kyc_hil_government_weekly.csv
-- Expected columns:
--   companyId, companyName, week_end_sunday, kyc_sessions_total
--   gov_validated, gov_not_validated
--   hil_doccheck_total, hil_doccheck_approved, hil_doccheck_disapproved
--   hil_liveness_total, hil_liveness_approved, hil_liveness_disapproved
--   hil_facematch_total, hil_facematch_approved, hil_facematch_disapproved

SELECT
    b.companyId,
    CONCAT('Company ', b.companyId) AS companyName,
    DATE_ADD(DATE(bs.createdAt), INTERVAL (6 - WEEKDAY(bs.createdAt)) DAY) AS week_end_sunday,

    COUNT(DISTINCT b.id) AS kyc_sessions_total,

    -- Government validation (document_check only)
    SUM(CASE WHEN bs.stepName = 'document_check' AND dcs.entityResponse = 1 THEN 1 ELSE 0 END) AS gov_validated,
    SUM(CASE WHEN bs.stepName = 'document_check' AND dcs.entityResponse = 0 THEN 1 ELSE 0 END) AS gov_not_validated,

    -- HIL by step: document_check
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'document_check' THEN 1 ELSE 0 END) AS hil_doccheck_total,
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'document_check' AND bs.status = 'Approved' THEN 1 ELSE 0 END) AS hil_doccheck_approved,
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'document_check' AND bs.status = 'Disapproved' THEN 1 ELSE 0 END) AS hil_doccheck_disapproved,

    -- HIL by step: liveness
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'liveness' THEN 1 ELSE 0 END) AS hil_liveness_total,
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'liveness' AND bs.status = 'Approved' THEN 1 ELSE 0 END) AS hil_liveness_approved,
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'liveness' AND bs.status = 'Disapproved' THEN 1 ELSE 0 END) AS hil_liveness_disapproved,

    -- HIL by step: facematch
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'facematch' THEN 1 ELSE 0 END) AS hil_facematch_total,
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'facematch' AND bs.status = 'Approved' THEN 1 ELSE 0 END) AS hil_facematch_approved,
    SUM(CASE WHEN bs.enableHumanInLoop = 1 AND bs.stepName = 'facematch' AND bs.status = 'Disapproved' THEN 1 ELSE 0 END) AS hil_facematch_disapproved
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
