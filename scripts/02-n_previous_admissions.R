source("scripts/utils.R")

conn <- connect()

q <- "
SELECT 
	cohort.admission_id
	, sum(CASE WHEN admissions.person_id IS NOT NULL THEN 1 ELSE 0 END) AS n_previous_admissions
FROM @cohort_schema.@cohort_table
LEFT JOIN derived_tables.admissions
	ON admissions.person_id = cohort.person_id
	AND admissions.discharge_datetime 
		BETWEEN cohort.admission_datetime - INTERVAL '@lookback years'
		AND cohort.admission_datetime
GROUP BY cohort.admission_id;
"
sql_fetch(q, conn,
		  cohort_schema = snakemake@params$cohort_schema,
		  cohort_table = snakemake@params$cohort_table,
		  lookback =snakemake@params$lookback) %>% 
	fwrite(snakemake@output[[1]], row.names = FALSE, sep = "\t")

dbDisconnect(conn)
