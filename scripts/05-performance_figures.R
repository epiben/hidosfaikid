source("scripts/utils.R")
library(ROCR)

this_study <- with(snakemake@wildcards, sprintf("%s__%s__%s__final", outcome_type, model_type, outcome_variable))

conn <- connect()
preds <- sql_fetch("SELECT dataset, pred_type, y_pred, y_true 
				    FROM @schema.predictions 
				    WHERE study_name = '@study_name' AND pred_type = 'crude';", 
				   conn, 
				   schema = snakemake@params$dbschema,
				   study_name = this_study)
dbDisconnect(conn)

calibration_plots <- preds %>% 
	group_by(dataset, pred_type) %>% 
	mutate(q = ntile(y_pred, 10),
		   dataset = factor(dataset, c("dev", "test", "test_new"), c("Development", "Test", "Test (unseen during training)")),
		   pred_type = factor(pred_type, c("crude", "platt", "isotonic"), c("Original", "Platt recalibration", "Isotonic recalibration"))) %>% 
	group_by(q, .add = TRUE) %>% 
	summarise(y_true_hi = qbeta(0.975, 0.5 + sum(y_true), 0.5 + sum(y_true == 0)), 
			  y_true_lo = qbeta(0.025, 0.5 + sum(y_true), 0.5 + sum(y_true == 0)), 
			  y_pred = mean(y_pred),
			  y_true = mean(y_true)) %>% 
	ggplot(aes(x = y_pred, y = y_true)) +
		geom_abline(intercept = 0, slope = 1, linetype = 2, size = 0.3, colour = "grey50") +
		geom_line(size = 0.5, colour = "grey80") +
		geom_linerange(aes(ymin = y_true_lo, ymax = y_true_hi), size = 0.5, colour = "grey30") +
		geom_point(size = 2, shape = 18) +
		# facet_grid(pred_type ~ dataset) +
		facet_wrap(~ dataset) +
		labs(x = "Bin-wise mean prediction", y = "Observed event proportion with 95% Jeffrey intervals") +
		theme_minimal() +
		theme(panel.spacing = grid::unit(1, "cm")) 

# ROC curves are unaffected by recalibration
roc_curves <- filter(preds, pred_type == "crude") %>% 
	dlply("dataset", function(x) with(x, prediction(y_pred, y_true))) %>% 
	ldply(function(x) { 
		p <- performance(x, "tpr", "fpr")
		tibble(fpr = unlist(p@x.values), tpr = unlist(p@y.values))
	}) %>% 
	mutate(dataset = factor(dataset, c("dev", "test", "test_new"), c("Development", "Test", "Test (unseen during training)"))) %>% 
	ggplot(aes(fpr, tpr)) +
		geom_abline(intercept = 0, slope = 1, linetype = 2, size = 0.3) +
		geom_line() +
		# geom_point(size = 0.5) +
		labs(x = "False positive rate", y = "True positive rate") +
		facet_wrap(~ dataset) +
		theme_minimal() +
		theme(panel.spacing = grid::unit(1, "cm"))

save_plot(calibration_plots, snakemake@output$calibration_plots)
save_plot(roc_curves, snakemake@output$roc_curves)
