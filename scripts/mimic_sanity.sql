-- Sanity checks against the SQL Server copy of the MIMIC-IV demo.
-- Run: sqlcmd -S .\SQLEXPRESS -d mimic_iv_demo -E -i scripts/mimic_sanity.sql
SET NOCOUNT ON;

-- 1. The harm proxy label, reproduced in T-SQL. Should be 60 of 275 (21.8%).
WITH complications AS (
    SELECT DISTINCT d.hadm_id
    FROM hosp.diagnoses_icd AS d
    WHERE (d.icd_version = 10 AND (
              d.icd_code LIKE 'T8[0-8]%' OR d.icd_code LIKE 'Y6[2-9]%'
           OR d.icd_code LIKE 'Y7[0-9]%' OR d.icd_code LIKE 'Y8[0-4]%'))
       OR (d.icd_version = 9  AND (
              d.icd_code LIKE '99[6-9]%' OR d.icd_code LIKE 'E87[0-9]%'
           OR d.icd_code LIKE 'E9[34][0-9]%'))
)
SELECT 'complication-of-care label' AS metric,
       COUNT(DISTINCT c.hadm_id) AS admissions_flagged,
       (SELECT COUNT(*) FROM hosp.admissions) AS admissions_total,
       CAST(100.0 * COUNT(DISTINCT c.hadm_id)
            / (SELECT COUNT(*) FROM hosp.admissions) AS DECIMAL(5,1)) AS pct
FROM complications AS c;

-- 2. Specific antidotes actually administered, and when relative to admission.
SELECT e.medication,
       COUNT(*) AS administrations,
       COUNT(DISTINCT e.hadm_id) AS admissions
FROM hosp.emar AS e
WHERE e.event_txt IN ('Administered', 'Administered in Other Location', 'Started')
  AND (e.medication LIKE '%Naloxone%' OR e.medication LIKE '%Flumazenil%'
    OR e.medication LIKE '%Protamine%' OR e.medication LIKE '%Dantrolene%')
GROUP BY e.medication
ORDER BY administrations DESC;

-- 3. Unplanned ICU escalation: arriving in intensive care from a general ward
--    rather than from the ED, recovery, or another ICU. Should be 35, matching
--    StructuredDataWatcher.unplanned_icu_transfer on the same cohort.
--    The '%ICU%' pattern is not redundant: "Trauma SICU (TSICU)" never spells out
--    "Intensive Care", and omitting it silently loses a case.
WITH ordered AS (
    SELECT t.hadm_id, t.careunit, t.intime, t.eventtype,
           LAG(t.careunit) OVER (PARTITION BY t.hadm_id ORDER BY t.intime) AS previous_unit
    FROM hosp.transfers AS t
),
escalation AS (
    SELECT hadm_id
    FROM ordered
    WHERE (careunit LIKE '%ICU%' OR careunit LIKE '%Intensive Care%'
        OR careunit LIKE '%(CCU)%' OR careunit LIKE '%Stepdown%'
        OR careunit LIKE '%Intermediate%')
      AND eventtype <> 'admit'          -- a direct admission is not a deterioration
      AND previous_unit IS NOT NULL     -- nor is the first unit of the stay
      AND previous_unit NOT LIKE '%Emergency%'
      AND previous_unit NOT LIKE '%PACU%'
      AND previous_unit NOT LIKE '%Discharge Lounge%'
      AND previous_unit NOT LIKE '%ICU%'
      AND previous_unit NOT LIKE '%Intensive Care%'
      AND previous_unit NOT LIKE '%(CCU)%'
      AND previous_unit NOT LIKE '%Stepdown%'
      AND previous_unit NOT LIKE '%Intermediate%'
)
SELECT COUNT(DISTINCT hadm_id) AS admissions_with_unplanned_escalation FROM escalation;

-- 4. Coded reoperations, which is how a return to theatre shows up in claims data.
SELECT p.long_title, COUNT(*) AS occurrences
FROM hosp.procedures_icd AS pi
JOIN hosp.d_icd_procedures AS p
  ON pi.icd_code = p.icd_code AND pi.icd_version = p.icd_version
WHERE p.long_title LIKE '%Reopening of%'
   OR p.long_title LIKE '%Control Bleeding%'
   OR p.long_title LIKE '%Reclosure%'
GROUP BY p.long_title
ORDER BY occurrences DESC;

-- 5. Timestamps really are DATETIME2, so length of stay is a date difference.
SELECT TOP 5
       a.hadm_id,
       a.admission_type,
       DATEDIFF(HOUR, a.admittime, a.dischtime) / 24.0 AS los_days
FROM hosp.admissions AS a
ORDER BY los_days DESC;
