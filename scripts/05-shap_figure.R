source("scripts/utils.R")

this_study <- with(snakemake@wildcards, sprintf("%s__%s__%s__final", outcome_type, model_type, outcome_variable))

n_quantiles <- snakemake@params$n_quantiles %||% 5
band_height <- snakemake@params$band_height %||% 0.8

compute_dens <- function(sub_df, keys, lims, dens_res = 1000, var = "shap_value") {
	density(sub_df[[var]], from = lims[1], to = lims[2], n = dens_res) %>% 
		with(tibble(shap_value = x, dens = y/max(y)))
}

n18_labels <- fread(read_file(snakemake@input$icd10_classification), select = c("V1", "V2")) %>% 
	.[grepl("^N18", V1)] %>% 
	with(setNames(V2, V1)) %>% 
	c(D189 = "Chronic kidney disease, other") # needed due to preprocessing error (bug fixed, so will be redundant after next full pipeline run

atc_map <- fread(read_file(snakemake@input$atc_map), select = c("ATC.code", "ATC.level.name"),
			  col.names = c("atc", "drug_name")) %>% 
 	with(setNames(str_to_sentence(drug_name), atc))

conn <- connect()

shap_values <- sql_fetch("SELECT * FROM @schema.shap_values WHERE study_name = '@study_name' AND pred_type = 'crude';", conn,
						 schema = snakemake@params$dbschema,
						 study_name = this_study) %>% 
	select(-dataset, -study_name, -pred_type) %>% 
	select_if(~ !all(is.na(.))) # keep complete columns

feature_values <- sql_fetch(sprintf("SELECT * FROM %s.data_test", snakemake@params$dbschema), conn) %>% 
	select(-person_id, -r, -time_at_risk, -starts_with("daily_rate_")) %>% 
	pivot_longer(-"admission_id", names_to = "feature", values_to = "feature_value")
dbDisconnect(conn)

d <- pivot_longer(shap_values, -c("y_pred", "y_true", "base_value", "admission_id"), names_to = "feature", values_to = "shap_value") %>%
	inner_join(feature_values, by = c("admission_id", "feature")) %>% 
	mutate(feature_pretty = case_when(
			   	grepl("atc_level2_", feature) ~ paste("MED:", atc_map[str_sub(feature, -3)]),
				grepl("n18_diag_", feature) ~ paste("DX:", n18_labels[str_sub(feature, -4)]),
				grepl("icd10_chapter_", feature) ~ paste("DX:", icd10_chapters[str_remove_all(str_sub(feature, -2), "_")]),
				TRUE ~ paste("DEMO/CLIN:", str_to_sentence(str_replace_all(feature, c("^n_" = "No. ", "_" = " "))))),
		   feature_pretty = fct_rev(factor(feature_pretty))) %>% 
	group_by(feature_pretty) %>% 
	mutate(feature_is_binary = n_distinct(feature_value) == 2,
		   feature_value = (feature_value - min(feature_value)) / diff(range(feature_value)), # put in [0, 1]
		   feature_value_bin = ifelse(feature_is_binary, feature_value, ntile(feature_value, min(n_distinct(feature_value), n_quantiles)))) %>% 
	group_by(feature_pretty, feature_value_bin) %>% 
	mutate(mean_feature_value = mean(feature_value)) %>% 
	group_by(feature_pretty, mean_feature_value) %>% 
	group_modify(compute_dens, lims = range(.$shap_value)) %>% 
	mutate(tile_width = diff(range(shap_value)) / (n() - 1)) %>% 
	filter(dens >= 0.001) # leave out areas with seriously low density, avoid warped plot

p <- ggplot(d, aes(x = feature_pretty, y = shap_value, fill = mean_feature_value, alpha = dens, group = mean_feature_value)) +
	geom_tile(aes(width = band_height, height = tile_width), position = position_dodge(band_height)) +
	scale_alpha_identity() +
	scale_fill_gradient(low = "blue", high = "red", breaks = NULL, labels = NULL) +
	scale_y_continuous(labels = scales::percent) +
	theme(strip.background = element_rect(fill = "grey95", size = 0)) +
	labs(y = "Absolute change in risk", x = "") +
	coord_flip() +
	theme_minimal()

save_plot(p, snakemake@output$fpath)
