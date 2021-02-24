source("scripts/utils.R")

fu_time <- snakemake@params$fu_time

status("Loading files")
cohort <- with(snakemake@params, load_cohort(cohort_schema, cohort_table))
datetime_cols <- paste0(c("admission", "discharge", "index"), "_datetime")
cohort[, (datetime_cols) := lapply(.SD, as_datetime), .SDcols = datetime_cols]

egfr <- fread(snakemake@input$egfr)
egfr[, egfr_datetime := as_datetime(egfr_datetime)]

status("Deriving eras")
egfr_eras <- merge(egfr, cohort, by = "person_id", allow.cartesian = TRUE) %>% 
	.[egfr_datetime %within% interval(admission_datetime, index_datetime + days(fu_time))]
setorder(egfr_eras, admission_id, egfr_datetime)
egfr_eras[, egfr_is_low := egfr <= 30]
egfr_eras[, era_id := cumsum(replace_na(egfr_is_low != shift(egfr_is_low) | admission_id != shift(admission_id), TRUE))]
egfr_eras <- egfr_eras[, .(admission_id = admission_id[1],
			  			   person_id = person_id[1],
						   egfr_is_low = egfr_is_low[1],
						   los_hours = los_hours[1],
						   index_datetime = index_datetime[1],
						   era_start_datetime = min(egfr_datetime)),
					   by = "era_id"] 
egfr_eras[, era_end_datetime := coalesce(shift(era_start_datetime, type = "lead") - seconds(1), 
										 index_datetime + hours(min(los_hours - 24, 24 * fu_time))),
		  by = "admission_id"]
egfr_eras[, era_id := NULL]
egfr_eras <- egfr_eras[era_start_datetime <= index_datetime + hours(los_hours)]
egfr_eras[, era_end_datetime := pmin(era_end_datetime, index_datetime + hours(los_hours), 
									 index_datetime + days(fu_time))]

status("Writing output")
fwrite(egfr_eras, snakemake@output$egfr_eras, row.names = FALSE, sep = "\t")

status("Done")
