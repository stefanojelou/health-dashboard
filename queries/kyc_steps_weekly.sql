-- Weekly KYC steps by company, stage and status
-- Run in MySQL (database: identity)
-- Export to CSV: data/kyc_steps_weekly.csv
-- Expected columns: companyId, companyName, week_end_sunday, stepName, status, kyc_steps_count
-- Note: this query avoids cross-schema joins, so it works with identity-only access.

SELECT
    b.companyId,
    CONCAT('Company ', b.companyId) AS companyName,
    DATE_ADD(DATE(bs.createdAt), INTERVAL (6 - WEEKDAY(bs.createdAt)) DAY) AS week_end_sunday,
    bs.stepName,
    bs.status,
    COUNT(*) AS kyc_steps_count
FROM identity.biometric_steps bs
JOIN identity.biometrics b
    ON b.id = bs.biometricId
WHERE bs.createdAt >= '2025-01-01'
  AND b.companyId IS NOT NULL
  AND bs.stepName IN ('document_check', 'liveness', 'facematch')
GROUP BY
    b.companyId,
    companyName,
    week_end_sunday,
    bs.stepName,
    bs.status
ORDER BY
    b.companyId,
    week_end_sunday,
    bs.stepName,
    bs.status;
