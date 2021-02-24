source("scripts/utils.R")

log_file <- start_log("logs/03_create_datasets.log")

status("Defining helper functions")
make_ohe <- function(d, var, prefix) {
	mutate(d, dummy = 1) %>% 
		pivot_wider(names_from = !!sym(var), names_prefix = prefix, values_from = dummy, values_fill = list(dummy = 0),
					values_fn = list(make_int = as.integer))
}

left_join_ohe <- function(d_left, d_right, var, prefix, by_var = "admission_id") {
	d_right <- mutate(d_right, dummy = 1) %>% 
		pivot_wider(names_from = !!sym(var), names_prefix = prefix, values_from = dummy, values_fill = list(dummy = 0),
					values_fn = list(make_int = as.integer))
	left_join(d_left, d_right, by = by_var) %>% 
		mutate_at(names(d_right), replace_na, replace = 0)
}

status("Loading input files")
input_fnames <- str_subset(names(snakemake@input), "\\w+") # handle snakemake's duplicate unnamed elements
for (fname in input_fnames) {
	assign(fname, fread(snakemake@input[[fname]]))
}

status("Downloading cohort table from database")
cohort <- with(snakemake@params, load_cohort(cohort_schema, cohort_table))

status("Creating features")
features <- cohort %>% 
	transmute(admission_id = as.integer(admission_id),
			  person_id = as.integer(person_id),
			  sex = tolower(sex),
			  age_at_admission = interval(date_of_birth, date(admission_datetime)) / years(1),
			  hour_of_admission_cyclical = abs(12 - hour(admission_datetime)),
			  weekday_admission = tolower(paste(wday(admission_datetime, label = TRUE, abbr = TRUE)))) %>% 
	make_ohe("weekday_admission", "admitted_") %>% 
	make_ohe("sex", "sex_") %>% 
	left_join(n_previous_admissions, by = "admission_id") %>% 
	left_join(n_distinct_drugs_given_before_index, by = "admission_id") %>% 
	left_join_ohe(n18_dx_past_5_years, "n18_diag", "n18_diag_") %>% 
	left_join_ohe(atc_level2_given_before_index, "atc_level2", "atc_level2_") %>% 
	left_join_ohe(icd10_chapters_past_5_years, "icd10_chapter", "icd10_chapter_") %>% 
	left_join(elixhauser_scores, by = "admission_id") %>% 
	mutate_if(~ !is.character(.), ~ replace_na(., 0)) %>% 
	left_join(select(cohort, admission_id, index_datetime), by = "admission_id") # index_datetime needed for train-test split

rm(list = input_fnames) # needed to avoid ambuity in column selection below

status("Creating full dataset")
df <- fread(snakemake@input$outcome_variables) %>% 
	inner_join(features, by = "admission_id") %>% 
	mutate(set = if_else(date(index_datetime) < "2015-07-01", "dev", "test"), # puts about 80% in dev set
		   set = case_when(set == "dev" ~ "dev",
		   				   set == "test" & !person_id %in% person_id[set == "dev"] ~ "test_new",
		   				   TRUE ~ "test_seen")) %>% 
	select(admission_id, set, everything(), -index_datetime) %>% 
	mutate_at(vars(everything(), -set, -time_at_risk, -age_at_admission), as.integer) %>% 
	select(set, admission_id, person_id, r, time_at_risk, age_at_admission, hour_of_admission_cyclical, 
		   n_previous_admissions, 
		   n_distinct_drugs_before_index, 
		   sex_female, sex_male, 
		   starts_with("n18_diag_"),
		   starts_with("admitted_"), 
		   starts_with("atc_level2_"), 
		   starts_with("icd10_chapter_"),
		   starts_with("elixhauser_score")) %>% 
	mutate(daily_rate_not_zero = 1 * (r > 0),
		   daily_rate_geq_1 = 1 * ((r/time_at_risk) >= 1),
		   daily_rate_geq_2 = 1 * ((r/time_at_risk) >= 2),
		   daily_rate_geq_3 = 1 * ((r/time_at_risk) >= 3),
		   daily_rate_geq_5 = 1 * ((r/time_at_risk) >= 5)) %>% 
	mutate_at(vars(starts_with("daily_rate_")), as.integer)

pick_features <- function(d) 
	select(d, -set, -admission_id, -person_id, -r, -time_at_risk, -starts_with("daily_rate_")) 

pick_outcomes <- function(d) 
	select(d, admission_id, person_id, r, time_at_risk, starts_with("daily_rate"))

normalise_features <- function(d, norm_factors) {
	pick_features(d) %>% 
		scale(coalesce(norm_factors$centre, 0), 
			  coalesce(norm_factors$scale, 1)) %>% 
		as_tibble()
}

status("Splitting into dev, test and test_new datasets + normalisation")
d_dev <- filter(df, set == "dev") 
d_test <- filter(df, set %in% c("test_seen", "test_new"))
d_test_new <- filter(df, set == "test_new")

# Normalise only non-binary features
features_dev <- pick_features(d_dev)
norm_idx <- !apply(features_dev, 2, function(.) all(. %in% 0:1))
norm_factors <- list(centre = ifelse(norm_idx, apply(features_dev, 2, mean), NA),
					 scale = ifelse(norm_idx, apply(features_dev, 2, sd), NA))

for (d_name in c("d_dev", "d_test", "d_test_new")) {
	assign(d_name, bind_cols(pick_outcomes(get(d_name)), 
							 normalise_features(get(d_name), norm_factors)))
}

status("Writing to database")
conn <- connect()
schema <- snakemake@params$cohort_schema
write_to_database("data_dev", d_dev, schema, append = FALSE)
write_to_database("data_test", d_test, schema, append = FALSE)
write_to_database("data_test_new", d_test_new, schema, append = FALSE)
write_to_database("normalisation_factors", bind_rows(purrr::transpose(norm_factors), .id = "feature_name"), 
				  schema, append = FALSE)
dbDisconnect(conn)

status("Setting timestamp")
write_file(paste(Sys.time()), snakemake@output$tstamp, append = FALSE)

status("Done")
