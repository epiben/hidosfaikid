source("scripts/utils.R")

# Helper function
impute_invalid_hours <- function(x, seed) { # imputation by sampling from the empirical distribution
	valid_idx <- second(x) != 1 # defining characteristic of timestamps with valid hour values
	valid_values <- hour(x[valid_idx])
	n_samples <- length(x) - length(valid_values)
	set.seed(seed)
	mean_vector <- sample(valid_values, n_samples, replace = TRUE) 
	set.seed(seed)
	hour(x[!valid_idx]) <- rnorm(n = n_samples, 
								 mean = mean_vector, 
								 sd = density(valid_values, cut = TRUE)$bw) %>% 
		round() %>% 
		pmin(23) %>% # enforce bounds
		pmax(0) # idem
	return(x)
}

status("Parsing snakemake parameters")
adm_min_len <- snakemake@params$admission_min_length

status("Downloading eGFR from database")
conn <- connect()
egfr <- sql_fetch(snakemake@input$sql_egfr, conn,
				  schema = snakemake@params$dbschema_biochem, 
				  table_name = snakemake@params$dbtable_biochem) %>% 
	as.data.table()

status("Finding eligible admissions")
# Eligible admissions: 
# - must begin after 2006-01-01 and before 2016-06-01 (we anyway only follow for 30 days)
# - length-of-stay >= 25 hours
# - no other admissions in previous 30 days (washout)
# - age >= 18 on date of admission

q <- "
SELECT adms.*, person.date_of_birth, person.sex
FROM @schema.@table_name AS adms
INNER JOIN (SELECT person_id, date_of_birth, sex FROM cpr_dev.person) AS person
	ON person.person_id = adms.person_id
	AND (admission_datetime::date - date_of_birth) >= 18 * 365.25
WHERE admission_datetime >= '2006-01-01 00:00:00'
	AND discharge_datetime <= '2016-06-30 23:59:59'
	AND discharge_datetime - admission_datetime >= interval '@admission_min_length hours'
"
eligible_admissions <- sql_fetch(q, conn, 
								 schema = snakemake@params$dbschema_admissions,
								 table_name = snakemake@params$dbtable_admissions,
								 admission_min_length = adm_min_len) %>% 
	as.data.table()

setorder(eligible_admissions, person_id, admission_datetime, discharge_datetime)
eligible_admissions[, delta_days := interval(lag(discharge_datetime, default = ymd("1970-01-01")), admission_datetime) / days(1)]
eligible_admissions[, prev_patient := shift(person_id)]
eligible_admissions <- eligible_admissions[(delta_days >= snakemake@params$washout & person_id == prev_patient) | 
											   	person_id != prev_patient | 
											   	is.na(prev_patient)] # we don't lose patients, so that's good
eligible_admissions[, index_datetime := admission_datetime + hours(adm_min_len)]
eligible_admissions[, c("delta_days", "prev_patient") := NULL]
eligible_admissions[, admission_datetime := impute_invalid_hours(admission_datetime, snakemake@params$seed)]
eligible_admissions[, discharge_datetime := impute_invalid_hours(discharge_datetime, snakemake@params$seed)]

status("Finding admissions with low eGFR")
admissions_with_low_egfr <- egfr[egfr <= 30, c("person_id", "egfr_datetime")] %>% 
	merge(eligible_admissions, by = "person_id", all.y = TRUE, allow.cartesian = TRUE) %>% 
	.[egfr_datetime %within% interval(admission_datetime, admission_datetime + hours(adm_min_len))]
admissions_with_low_egfr[, "egfr_datetime" := NULL]
admissions_with_low_egfr <- unique(admissions_with_low_egfr)

status("Removing eGFR values for irrelevant patients")
egfr <- merge(egfr, admissions_with_low_egfr[, "person_id"], by = "person_id") # no need to keep all eGFR values

status("Saving output files")
fwrite(admissions_with_low_egfr, snakemake@output$adms_with_low_egfr, row.names = FALSE, sep = "\t")
fwrite(eligible_admissions, snakemake@output$eligible_adms, row.names = FALSE, sep = "\t")
fwrite(egfr, snakemake@output$egfr, row.names = FALSE, sep = "\t")

status("Done")
