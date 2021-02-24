source("scripts/utils.R")
library(rmda)

this_study <- with(snakemake@wildcards, sprintf("%s__%s__%s__final", outcome_type, model_type, outcome_variable))

conn <- connect()
q <- "
SELECT y_true, y_pred 
FROM @schema.predictions 
WHERE study_name = '@study_name'
	AND dataset = 'test'
	AND pred_type = 'crude'
"
preds <- sql_fetch(q, conn, schema = snakemake@params$dbschema, study_name = this_study)
dbDisconnect(conn)

p_data <- decision_curve(y_true ~ y_pred, data = preds, fitted.risk = TRUE, thresholds = 0:10/10) %>% 
	rmda:::preparePlotData(confidence.intervals = TRUE, curve.names = NA) %>% 
	with(dc.data) %>% 
	mutate(model = ifelse(model == "y_true ~ y_pred", "Prediction model", model))

x_axis <- distinct(p_data, b = thresholds, l = cost.benefit.ratio)
colour_map <- c("None" = "grey40", "All" = "70", "Prediction model" = "dodgerblue")
y_lim <- c(min(filter(p_data, model == "Prediction model")$sNB, na.rm = TRUE), NA)

p <- ggplot(p_data, aes(thresholds, sNB)) +
	geom_ribbon(aes(ymin = sNB_lower, ymax = sNB_upper, fill = model), size = 0, alpha = 0.1) +
	geom_line(aes(colour = model)) +
	theme_minimal() +
	scale_colour_manual(values = colour_map) +
	scale_fill_manual(values = colour_map) +
	theme(panel.grid.minor.x = element_blank(),
		  legend.title = element_blank()) +
	coord_cartesian(ylim = y_lim) +
	labs(x = "High-risk threshold", y = "Standardised net benefit") +
	scale_x_continuous(breaks = x_axis$b, labels = scales::percent, 
					   sec.axis = dup_axis(name = "Cost-benefit ratio", labels = x_axis$l))

save_plot(p, snakemake@output$fpath)
