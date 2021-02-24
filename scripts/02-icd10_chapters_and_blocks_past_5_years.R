source("scripts/utils.R")

log_file <- start_log("logs/02_icd10_chapters_blocks_past_5_years.log")

status("Loading ICD10 classification")
icd10_classification <- fread(read_file(snakemake@input$icd10_classification), select = c("V1", "V4", "V6"), 
							  col.names = c("diag_code", "icd10_chapter", "icd10_block"))
icd10_classification[, icd10_block := parse_integer(str_remove_all(icd10_block, "R"))]

status("Importing diagnoses from dbserver")
conn <- connect()
q <- "
SELECT DISTINCT 
	admission_id
	, SUBSTRING(diag_code, 2) AS diag_code -- remove D prefix
FROM @cohort_schema.@cohort_table
LEFT JOIN @lpr_schema.adm
	ON adm.person_id = cohort.person_id
	AND adm.departure_date BETWEEN admission_datetime::date - INTERVAL '5 years' AND admission_datetime::date
LEFT JOIN @lpr_schema.diag
	ON diag.visit_id = adm.visit_id;  
"
diagnoses <- sql_fetch(q, conn,
					   lpr_schema = snakemake@params$lpr_schema,
					   cohort_schema = snakemake@params$cohort_schema,
					   cohort_table = snakemake@params$cohort_table) %>% 
	merge(icd10_classification, by = "diag_code")
dbDisconnect(conn)

# should really check if some can't be mapped and decide what to do

status("Making output tables")
icd10_chapters <- unique(diagnoses[, c("admission_id", "icd10_chapter")])
icd10_blocks <- unique(diagnoses[, c("admission_id", "icd10_block")])

status("Writing output files")
fwrite(icd10_blocks, snakemake@output$icd10_blocks, row.names = FALSE, sep = "\t")
fwrite(icd10_chapters, snakemake@output$icd10_chapters, row.names = FALSE, sep = "\t")

status("Done")