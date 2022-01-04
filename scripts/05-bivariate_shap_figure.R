source("scripts/utils.R")
library(patchwork)
library(purrr)

status("Downloading from server and wrangling")
features <- snakemake@params$features %>% 
	lapply(function(x) sprintf("\"%s\"", x)) %>% # to handle upper-case letters
	paste(collapse = ", ") 

q <- "SELECT admission_id, study_name, @features FROM @schema.shap_values 
	  WHERE pred_type = 'crude' 
	      AND dataset = 'test'
		  AND NOT study_name ~ 'refmodel_';"

conn <- connect()

shap_values <- sql_fetch(q, conn, 
						 features = features,
						 schema = snakemake@params$dbschema) %>% 
	mutate(study_name = str_replace_all(study_name, c("binary__|__final" = "",
													  "mlp" = "MLP",
													  "linear" = "Linear",
													  "geq" = ">=",
													  "not_zero" = ">0",
													  "__" = ", ",
													  "_" = " "))) 

norm_factors <- sql_fetch("SELECT * FROM @schema.normalisation_factors;", conn, # centre and scale are NA for binary features
						   schema = snakemake@params$dbschema)

feature_values <- sql_fetch("SELECT * FROM @schema.data_test", conn,
							schema = snakemake@params$dbschema) %>% 
	select(-person_id, -r, -time_at_risk, -starts_with("daily_rate_")) %>% 
	pivot_longer(-"admission_id", names_to = "feature", values_to = "feature_value") %>% 
	left_join(rename(norm_factors, feature = feature_name), by = "feature") %>% 
	mutate(feature_value = if_else(is.na(centre) | is.na(scale), feature_value, round(feature_value * scale + centre, 8))) %>% 
	select(-scale, -centre)

dbDisconnect(conn)

shap_df <- pivot_longer(shap_values, -c("study_name", "admission_id"), names_to = "feature", values_to = "shap_value") %>% 
	inner_join(feature_values, by = c("admission_id", "feature")) %>% 
	separate(study_name, into = c("model", "outcome"), sep = ", ") %>% 
	mutate(colour = parse_number(str_sub(outcome, end = -1))) %>% 
	arrange(parse_number(str_sub(outcome, end = -1)), model) %>% 
	mutate(outcome = fct_inorder(factor(outcome)),
		   group = fct_inorder(factor(paste(outcome, model))),
		   feature = str_replace_all(feature, c("n_" = "No. ", "icd" = "ICD", "atc" = "ATC", "_" = " ")),
		   feature = paste0(toupper(str_sub(feature, 1, 1)), str_sub(feature, 2))) # plot cosmetics

bivariate_shap_plot <- function(f, df = shap_df, sample_size = NULL, band_height = 0.8) {
	compute_dens <- function(sub_df, keys, lims, dens_res = 512, var = "shap_value") {
		density(sub_df[[var]], n = dens_res) %>% 
			with(tibble(shap_value = x, dens = y/max(y)))
	}
	
	set.seed(42)
	df <- filter(df, 
				 feature == f,
				 feature_value <= quantile(feature_value, snakemake@params$quantile_cutoff %||% 0.999))
	try(df <- sample_n(df, sample_size), silent = TRUE)
	
	if (all(df$feature_value %in% 0:1)) {
		d <- group_by(df, feature, model, outcome, feature_value, group) %>% 
			group_modify(compute_dens) %>% 
			mutate(tile_width = diff(range(shap_value)) / (n() - 1),
				   feature_value = factor(feature_value, 0:1, c("No", "Yes"))) %>% 
			ungroup()
		
		line_df <- group_by(d, feature, model, outcome, feature_value, group) %>% 
			summarise(y_min = min(shap_value), y_max = max(shap_value), .groups = "drop")
		
		p <- ggplot(d, aes(x = feature_value, group = group)) + 
			geom_hline(yintercept = 0, size = 0.25, linetype = 2) +
			geom_linerange(aes(ymin = y_min, ymax = y_max, linetype = model), line_df, position = position_dodge(band_height),
						   size = 0.5, colour = "grey60") +
			geom_tile(aes(y = shap_value, alpha = dens, fill = outcome, width = band_height*0.6, height = tile_width), 
					  position = position_dodge(band_height)) +
			scale_alpha_identity() +
			scale_fill_brewer(palette = "Set1") +
			guides(linetype = FALSE, fill = FALSE) + # requires there be at least one continuous variable
			theme_minimal() +
			theme(panel.grid.major.x = element_blank())
	} else {
		p <- ggplot(df, aes(x = feature_value, y = shap_value))+
			geom_hline(yintercept = 0, size = 0.25, linetype = 2) +
			geom_smooth(aes(colour = outcome, linetype = model), size = 0.5, method = "loess", 
						se = FALSE, formula = y ~ x) +
			scale_colour_brewer(palette = "Set1") +
			theme_minimal() 
	}
	
	p + 
		facet_wrap(~ feature, scales = "free") +
		scale_linetype_manual(values = c("MLP" = "solid", "Linear" = "longdash")) +
		scale_y_continuous(labels = scales::percent) +
		labs(x = "Feature value (original scale)", y = "Shap value") +
		theme(strip.background = element_rect(fill = "grey97", size = 0))
}

status("Building plots")
plots <- sapply(unique(shap_df$feature), bivariate_shap_plot, sample_size = snakemake@params$sample_size, 
				USE.NAMES = TRUE, simplify = FALSE) %>% 
	lapply(ggplot_build)

p <- lapply(plots, with, plot) %>% 
	reduce(`%+%`) + 
	plot_layout(guides = "collect", ncol = 3)

status("Saving plots and ggbuild object")
save_plot(p, snakemake@output$plot_fname)
write_rds(plots, snakemake@output$plot_ggbuild)

status("Done")
