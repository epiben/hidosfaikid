source("scripts/utils.R")

status("Downloading from dbserver and computing scores")
conn <- connect()
q <- "
SELECT 
	admission_id
	, SUBSTRING(diag_code, 2) as diag_code
FROM @cohort_schema.@cohort_table AS cohort
LEFT JOIN @lpr_schema.adm
	ON adm.person_id = cohort.person_id
	AND adm.departure_date < cohort.admission_datetime::date
LEFT JOIN @lpr_schema.diag
	ON diag.visit_id = adm.visit_id
"
scores <- sql_fetch(q, conn,
					  lpr_schema = snakemake@params$lpr_schema,
					  cohort_schema = snakemake@params$cohort_schema,
					  cohort_table = snakemake@params$cohort_table) %>% 
	comorbidity::comorbidity("admission_id", "diag_code", "elixhauser", assign0 = FALSE, "icd10") %>% 
	select(admission_id, elixhauser_score_ahrq = wscore_ahrq) %>% 
	mutate_at("elixhauser_score_ahrq", replace_na, replace = 0) # fair assumption (manually checked)
dbDisconnect(conn)

status("Writing output file")
fwrite(scores, snakemake@output$elixhauser_scores, row.names = FALSE, sep = "\t")

status("Done")
