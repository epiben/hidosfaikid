source("scripts/utils.R")
library(tableone)

conn <- connect()
norm_factors <- sql_fetch("SELECT * FROM @schema.normalisation_factors;", conn, schema = snakemake@params$dbschema)
	# centre and scale are NA for binary features

target_drugs_given <- read_tsv(snakemake@input$outcome_variables) %>% 
	select(admission_id, starts_with("r_")) %>% 
	mutate_at(vars(-admission_id), ~ 1 * (. > 0)) %>% 
	rename_at(which(grepl("r_", names(.))), ~ paste0("inapp_dose_of_", toupper(str_sub(., 3))))

df <- sapply(c("dev", "test", "test_new"), 
			 function(.) sql_fetch(sprintf("SELECT * FROM %s.data_%s;", snakemake@params$dbschema, .), conn), 
			 USE.NAMES = TRUE, simplify = FALSE) %>% 
	bind_rows(.id = "set") %>% 
	pivot_longer(-c(set:daily_rate_geq_5), names_to = "feature_name", values_to = "feature_value") %>% 
	left_join(norm_factors, by = "feature_name") %>% 
	mutate(feature_value = if_else(is.na(centre) | is.na(scale), feature_value, round(feature_value * scale + centre, 8))) %>% 
	select(-scale, -centre) %>% 
	pivot_wider(names_from = "feature_name", values_from = "feature_value") %>% 
	inner_join(target_drugs_given, by = "admission_id") %>% 
	select(-admission_id)
dbDisconnect(conn)

table1_df <- df %>% 
	group_by(set) %>% 
	mutate(n_persons = factor(1 * (!duplicated(person_id))),
		   n_women = factor(1 * (sex_female == 1 & !duplicated(person_id)))) %>% 
	select(-person_id) %>% 
	ungroup() %>% 
	mutate(n18_diag_any = apply(select(., starts_with("n18_diag")), 1, max)) %>% 
	mutate_at(vars(matches("^(r$|n_diag|n_distinct_|elixhauser_|n_previous_admissions)")), 
			  cut, breaks = c(-Inf, 0, 2, 4, 6, Inf), labels = c("0", "1-2", "3-4", "5-6", ">6")) %>% 
	mutate_at(vars(matches("^(inapp_dose|sex|admitted|atc_level2|n18_|daily_rate_|icd10_chapter)")), factor) %>% 
	mutate_at("hour_of_admission_cyclical", cut, breaks = c(-Inf, 0:4 * 3), labels = c("0", "1-3", "4-6", "7-9", "10-12")) 
		# NB! This feature only goes to 12
	
table1 <- CreateTableOne(strata = "set", data = table1_df, test = FALSE) %>% 
	print(contDigits = 1, nonnormal = c("time_at_risk", "age_at_admission", "n_previous_admissions"), 
		  dropEqual = TRUE, printToggle = FALSE) 

table1 <- bind_cols("variate" = row.names(table1), as_tibble(table1, .name_repair = ~ c("develop", "test", "test_new"))) %>% 
	mutate_at(-1, ~ str_trim(str_replace_all(., "\\(\\s+", "("))) 

# Brute force but it works and takes no time (replace indentation with "group label"
prev_label <- ""
for (i in seq_along(table1$variate)) {
	v <- table1$variate[i]
	if (str_sub(v, 1, 3) == "   ") {
		table1$variate[i] <- sprintf("%s: %s", prev_label, str_sub(v, 4))
	} else {
		prev_label <- v
	}
}

# table1 is actually too comprehensive and so will be in the supplement whereas the actual table 1 will be a subset
in_table1_regex <- "^n$|^n_|^daily_rate|time_at_risk|^inapp_dose|^admitted|_N02|_J01|_C03|_B01|_A02|n18_diag_any|chapter_(4|9|14|18|19)|unique_person|sex_female"

table1 <- table1 %>% 
	mutate(variate = fct_inorder(factor(variate)),
		   i = case_when(grepl("atc_level2", variate) ~ 3,
						 grepl("icd10_chapter", variate) ~ 2,
						 grepl("^n$|^r |^inapp_dose|time_at_risk|daily_rate", variate) ~ 0,
						 TRUE ~ 1),
		   in_table1 = ifelse(str_detect(variate, in_table1_regex), "yes", "no")) %>% 
	arrange(desc(in_table1), i, variate) %>% 
	select(-i) %>% 
	filter(develop != "", !str_detect(variate, "^set \\(\\%\\)"))

write_tsv(table1, snakemake@output$fpath, append = FALSE)
