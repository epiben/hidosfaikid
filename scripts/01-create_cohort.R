source("scripts/utils.R")

status("Loading files")
admissions_with_low_egfr <- fread(snakemake@input$adms_with_low_egfr)
# eligible_admissions <- fread(snakemake@input$eligible_admissions)

datetime_cols <- paste0(c("admission", "discharge", "index"), "_datetime")
admissions_with_low_egfr[, (datetime_cols) := lapply(.SD, as_datetime), .SDcols = datetime_cols]
# eligible_admissions[, (datetime_cols) := lapply(.SD, as_datetime), .SDcols = datetime_cols]

# conn <- connect()
# medication_eras <- sql_fetch("SELECT * FROM @schema.medication_eras WHERE atc_code ~ '@atc_regex';", conn,
# 							 schema = snakemake@params$cohort_schema,
# 							 atc_regex = sprintf("^(%s)", paste(snakemake@params$target_atc, collapse = "|")))
# dbDisconnect(conn)

status("Finding cohort")

# cohort <- medication_eras %>% 
# 	merge(admissions_with_low_egfr, by = "person_id", allow.cartesian = TRUE) %>% 
# 	.[int_overlaps(prescr_start %--% prescr_end, admission_datetime %--% (admission_datetime + hours(25)))] %>% 
# 	# .[int_overlaps(prescr_start %--% prescr_end, admission_datetime %--% (discharge_datetime))]
# 	merge(eligible_admissions, by = "admission_id")
cohort <- admissions_with_low_egfr
cohort[, los_hours := as.integer(interval(admission_datetime, discharge_datetime) / hours(1))]
cohort[, date_of_birth := as_date(date_of_birth)]

status("Saving cohort table in database")
conn <- connect()
write_to_database(snakemake@params$cohort_table, cohort, snakemake@params$cohort_schema, append = FALSE)

status("Creating admission_id-visit_id map in database")
q <- "
BEGIN TRANSACTION;
SET SEARCH_PATH TO @cohort_schema;
DROP TABLE IF EXISTS visit_lookup;
SELECT cohort.admission_id, vis.visit_id
INTO visit_lookup
FROM @cohort_schema.@cohort_table
INNER JOIN derived_tables.admission_visits AS vis
	ON vis.admission_id = cohort.admission_id;
COMMIT;
"
sql_exec(q, conn,
		 cohort_schema = snakemake@params$cohort_schema,
		 cohort_table = snakemake@params$cohort_table)
dbDisconnect(conn)

status("Writing timestamp")
write_file(paste(Sys.time()), snakemake@output$tstamp, append = FALSE)

status("Done")
