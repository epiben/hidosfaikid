source("scripts/utils.R")

this_study <- with(snakemake@wildcards, sprintf("%s__%s__%s", outcome_type, model_type, outcome_variable))

beautify_names <- function(x) {
	str_split(x, pattern = "__") %>% 
		lapply(str_replace_all, pattern = c("binary" = "Binary", "poisson" = "Poisson", "zip" = "ZIP",
											"mlp" = "MLP", "geq" = ">=", "not_zero" = "> 0", "_" = " ")) %>% 
		lapply(function(s) sprintf("%s %s \n(%s)", s[1], s[2], s[3])) %>% 
		unlist()
}

conn <- connect()
q <- "
SET search_path TO @schema;
SELECT 
	trials.trial_id
	, trials.number AS trial_number
	, trial_values.value AS metric_value
	, study_name
	, param_name
	, param_value
	, CASE 
		WHEN distribution_json LIKE '%Categorical%' THEN (distribution_json::json#>'{attributes,choices}')::json->>(param_value::int)
		ELSE param_value::text
		END AS real_param_value
FROM trial_params AS params
INNER JOIN trials
	ON trials.trial_id = params.trial_id
INNER JOIN trial_values
	ON trial_values.trial_id = params.trial_id
INNER JOIN studies
	ON studies.study_id = trials.study_id
WHERE study_name = '@study_name';
"
# Warnings are fine, they're by design
param_values <- as_tibble(sql_fetch(q, conn, 
									schema = snakemake@params$dbschema,
									study_name = this_study)) %>% 
	mutate(real_param_value_numeric = parse_number(real_param_value),
		   real_param_value_string = if_else(is.na(real_param_value_numeric), real_param_value, NA_character_),
		   study_name_pretty = beautify_names(this_study))

best_trial_id <- sql_fetch("SELECT trial_id 
							FROM @schema.best_trials
							WHERE study_name = '@study_name'
						    ORDER BY trial_id DESC
						    LIMIT 1", conn,
							schema = snakemake@params$dbschema,
							study_name = this_study) %>% 
	as.numeric()
dbDisconnect(conn)

best_trial_number <- unique(filter(param_values, trial_id == best_trial_id)$trial_number)

draw_plot <- function(d) {
	param_is_number <- !any(is.na(d$real_param_value_numeric))
	y_var <- paste0("real_param_value_", if (isTRUE(param_is_number)) "numeric" else "string")
	if (isTRUE(param_is_number)) {
		d <- mutate(d, new_y_var = !!sym(y_var))
	} else {
		d <- mutate(d, new_y_var_label = str_replace_all(!!sym(y_var), "_", " "),
					new_y_var = as.numeric(factor(new_y_var_label)))
	}
	
	p75 <- with(d, metric_value < quantile(metric_value, 0.75, na.rm = TRUE))
	
	p <- ggplot(d, aes(x = trial_number, y = new_y_var)) +
		geom_vline(xintercept = best_trial_number, linetype = 2, size = 0.3, colour = "grey40") +
		geom_hline(aes(yintercept = new_y_var), ~ filter(., trial_number == best_trial_number), 
				   linetype = 2, colour = "grey40", size = 0.3) +
		geom_line(colour = "grey80", size = 0.5) +
		geom_point(data = ~ filter(., metric_value > p75), colour = "grey50", size = 1) +
		geom_point(data = ~ filter(., is.na(metric_value)), colour = "grey50", shape = 18, size = 1) +
		geom_point(aes(colour = metric_value), ~ filter(., metric_value <= p75), size = 1, show.legend = FALSE) +
		labs(x = "Optuna trial (\"timeline\")") +
		scale_colour_gradient(high = "red", low = "dodgerblue") +
		facet_wrap(~ param_name) + # cosmetics
		theme_minimal() +
		theme(axis.title = element_blank(),
			  panel.grid.major.y = element_line(colour = "grey95"),
			  panel.grid.major.x = element_blank(),
			  panel.grid.minor = element_blank())
	if (isTRUE(param_is_number)) {
		p + scale_y_continuous(breaks = c(10^(-(15:1)), 2^(0:10)), trans = "log2")
	} else {
		y <- distinct(d, new_y_var, new_y_var_label)
		p + scale_y_continuous(p, breaks = y$new_y_var, labels = y$new_y_var_label)
	}
}

p <- dlply(param_values, "param_name", draw_plot) %>% 
	ggpubr::ggarrange(plotlist = ., ncol = 2, nrow = ceiling(length(.)/2))

p
status("Writing outputs")
save_plot(p, snakemake@output$fpath)

# Old code with all studies in one plot for overview
# # Precompute offsets so they stay the same throughout the timeline
# offsets <- with(param_values, setNames(seq(-0.05, 0.05, length.out = n_distinct(param_name)), unique(param_name)))
# 
# plot_df_full <- group_by(param_values, study_name, param_name) %>% 
# 	arrange(study_name, param_name) %>% 
# 	mutate(param_value = (param_value - min(param_value)) / diff(range(param_value))) %>% 
# 	group_by(study_name) %>% 
# 	mutate(param_value = param_value + offsets[param_name],
# 		   study_name_pretty = beautify_names(study_name))
# 
# # Supp. figure (all traces)
# p_full <- ggplot(plot_df_full, aes(x = trial_number, y = param_value, colour = param_name)) +
# 	geom_line(alpha = 0.7, size = 0.5) +
# 	geom_point(size = 0.4) +
# 	facet_grid(study_name_pretty ~ ., switch = "y") +
# 	labs(x = "Optuna trial (\"timeline\")") +
# 	scale_x_continuous(trans = "log2", breaks = c(2^(0:6), 100)) +
# 	theme_minimal() +
# 	theme(strip.text.y.left = element_text(angle = 0),
# 		  axis.text.y = element_blank(),
# 		  axis.title.y = element_blank(),
# 		  legend.position = "bottom",
# 		  legend.title = element_blank())
