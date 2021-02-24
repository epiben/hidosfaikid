source("scripts/utils.R")

status("Loading input files")
cohort <- with(snakemake@params, load_cohort(cohort_schema, cohort_table))

drug_admins <- fread(snakemake@input$all_drug_admins, select = c("person_id", "admin_datetime", "atc")) %>% 
	.[grep("^\\w\\d{2}\\w{2}\\d{2}", atc)] # only full ATC code of valid format
drug_admins[, admin_datetime := as_datetime(admin_datetime)]

status("Wrangling")
o <- merge(drug_admins, cohort, by = "person_id", all.y = TRUE, allow.cartesian = TRUE) %>% 
	.[admin_datetime %within% interval(admission_datetime, index_datetime)] %>% 
	.[, .(n_distinct_drugs_before_index = n_distinct(atc, na.rm = TRUE)), by = "admission_id"]

status("Writing output file")
fwrite(o, snakemake@output$fpath, row.names = FALSE, sep = "\t")

status("Done")
