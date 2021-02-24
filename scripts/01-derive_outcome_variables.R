source("scripts/utils.R")

status("Loading input files")
cohort <- with(snakemake@params, load_cohort(cohort_schema, cohort_table))

egfr_eras <- fread(snakemake@input$egfr_eras)
target_drug_admins <- fread(snakemake@input$target_drug_admins)

status("Giving timestamp correct format")
target_drug_admins[, admin_datetime := as_datetime(admin_datetime)]

datetime_cols <- paste0(c("index", "era_start", "era_end"), "_datetime")
egfr_eras[, (datetime_cols) := lapply(.SD, as_datetime), .SDcols = datetime_cols]

status("Computing time at risk")
time_at_risk <- egfr_eras %>% 
	.[(egfr_is_low), 
	  .(time_at_risk = min(snakemake@params$fu_time, 
	  					 sum(as.numeric(era_end_datetime - era_start_datetime, "days")))),
	  by = "admission_id"]

status("Computing number of administrations with inappropriate doses")
# First, apixaban because of particular logic
r_apixaban <- select(target_drug_admins, person_id, admin_datetime, atc, given_dose) %>% 
	filter(grepl("^B01AF02", atc)) %>% # apixaban
	inner_join(filter(egfr_eras, egfr_is_low), by = "person_id") %>% 
	filter(admin_datetime %within% interval(era_start_datetime, era_end_datetime)) %>% 
	mutate(admin_date = date(admin_datetime)) %>% 
	group_by(admission_id, admin_date) %>% 
	arrange(admin_datetime, .by_group = TRUE) %>% 
	mutate(cum_daily_dose = cumsum(given_dose)) %>% 
	filter(cum_daily_dose > 5) %>% 
	group_by(admission_id, atc) %>% 
	summarise(r = n(), .groups = "drop")

# Then, the rest (apixaban added approx. mid-chain)
outcome <- filter(egfr_eras, egfr_is_low) %>% 
	inner_join(select(target_drug_admins, person_id, admin_datetime, atc), by = "person_id") %>% 
	filter(!grepl("^B01AF02", atc)) %>%  # apixaban
	filter(admin_datetime %within% interval(era_start_datetime, era_end_datetime)) %>% 
	group_by(admission_id, atc) %>% 
	summarise(r = n(), .groups = "drop") %>% 
	bind_rows(r_apixaban) %>% 
	mutate(atc = paste0("r_", tolower(atc))) %>% 
	group_by(admission_id) %>% 
	bind_rows(summarise(., atc = "r", r = sum(r))) %>% 
	pivot_wider(admission_id, names_from = atc, values_from = r, values_fill = list(r = 0)) %>% 
	right_join(time_at_risk, by = "admission_id") %>% 
	mutate_all(replace_na, 0) # for those without any administrations

status("Writing output")
fwrite(outcome, snakemake@output$outcome_variables, row.names = FALSE, sep = "\t")

status("Done")
