source("scripts/utils.R")

status("Loading input files")
cohort <- with(snakemake@params, load_cohort(cohort_schema, cohort_table))

drug_admins <- fread(snakemake@input$all_drug_admins, select = c("person_id", "admin_datetime", "atc")) %>% 
	.[grep("^\\w\\d{2}\\w{2}\\d{2}", atc)] # only full ATC code of valid format
drug_admins[, ':='(admin_datetime = as_datetime(admin_datetime),
				   atc_level2 = str_sub(atc, 1, 3),
				   atc = NULL)]

status("Wrangling")
o <- merge(drug_admins, cohort, by = "person_id", all.y = TRUE, allow.cartesian = TRUE) %>% 
	.[admin_datetime %within% interval(admission_datetime, index_datetime)] %>% 
	.[, c("admission_id", "atc_level2")] %>% 
	unique() %>% 
	merge(.[, .N, by = "atc_level2"][N >= snakemake@params$min_patient_per_atc, "atc_level2"], by = "atc_level2")

status("Writing output")
fwrite(o, snakemake@output$fpath, row.names = FALSE, sep = "\t")

status("Done")
