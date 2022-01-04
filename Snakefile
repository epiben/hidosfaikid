# Authors: 
# 	Benjamin Skov Kaas-Hansen (creator, maintainer) 
# 	Davide Placido (contributor)
# 	Cristina Leal Rodríguez (contributor)

# ========== SETTINGS =========

MODEL_CONFIGS = [ # a little longer but much more readable than expand()
	"binary__mlp__daily_rate_not_zero", "binary__linear__daily_rate_not_zero",
	"binary__mlp__daily_rate_geq_1", "binary__linear__daily_rate_geq_1", 
	"binary__mlp__daily_rate_geq_2", "binary__linear__daily_rate_geq_2", 
	"binary__mlp__daily_rate_geq_3", "binary__linear__daily_rate_geq_3", 
	"binary__mlp__daily_rate_geq_5", "binary__linear__daily_rate_geq_5",
	"refmodel__mlp__daily_rate_not_zero", "refmodel__linear__daily_rate_not_zero",
	"refmodel__mlp__daily_rate_geq_1", "refmodel__linear__daily_rate_geq_1"
]

TARGET_DRUGS = [
	"B01AF02", # Apixaban, daily dose <= 5 mg. when eGFR <= 30
	# "B01AF03", # Edoxaban, there are no patients using this anyway
	"B01AE07", # Dabigatran,
	"B01AF01", # Rivaroxaban
	"A10BA02", # Metformin
	"M01AE01", # Ibuprofene
	"M01AH01", # Celecoxib
	"L04AX03", # Methotrexate
	"N05AN01" # Lithium citrate
]

BIVARIATE_SHAP_FEATURES = [
	"age_at_admission", 
	"n_previous_admissions", 
	"n_distinct_drugs_before_index",
	"n18_diag_N182", 
	"icd10_chapter_14", 
	"icd10_chapter_4", 
	"atc_level2_N02", 
	"atc_level2_A10",
	"atc_level2_C03"
]

shell.prefix("""
module purge
module load tools
module load anaconda3/2021.05
module load gcc/6.2.0 intel/redist/2017.2.174 intel/perflibs/64/2020_update2 
module load R/3.6.1-GCC-MKL
""")

# ========== 00 TARGET RULE ==========
rule target:
	input: 
		"data/05-table_1_summary_stats.tsv",
		"data/05-bivariate_shap_plot.pdf",
		expand("data/04-final/hist_final__{x}.tsv", x=MODEL_CONFIGS),
		expand("data/05-optuna_learning/{x}.pdf", x=MODEL_CONFIGS),
		expand("data/05-performance/calibration_plots__{x}.pdf", x=MODEL_CONFIGS),
		expand("data/05-shap/{x}.pdf", x=MODEL_CONFIGS),
		expand("data/05-decision_curves/{x}.pdf", x=MODEL_CONFIGS),
		"data/05-snakemake_rulegraph.pdf",
		"data/06-supplement.pdf"

# ========== 01 DATA WRANGLING ==========		
rule fetch_egfr_and_admissions_with_low_egfr: 
	input: sql_egfr = "scripts/01-fetch_valid_egfr.sql"
	output: 
		egfr = "data/01-egfr.tsv",
		adms_with_low_egfr = "data/01-admissions_with_low_egfr.tsv",
		eligible_adms = "data/01-eligible_admissions.tsv"
	params:
		dbname = "bth",
		dbuser = "benkaa",
		dbschema_biochem = "temp_biochem",
		dbtable_biochem = "biochem",
		dbschema_admissions = "derived_tables",
		dbtable_admissions = "admissions",
		washout = 30, # days
		admission_min_length = 25, # hours
		seed = 42
	benchmark: "benchmarks/01-fetch_egfr_and_admissions_with_low_egfr.tsv"
	resources: vmem = 1024*10, tmin = 60
	script: "scripts/01-fetch_egfr_and_find_admissions_with_low_egfr.R"

rule fetch_drug_administrations: 
	input:
		epm3_header = "data/00-raw-symlinks/epm3_header.tsv",
		epm3_admins = "data/00-raw-symlinks/epm3_administrations.tsv",
		epm1_admins_header = "data/00-raw-symlinks/epm1_admins_header.tsv",
		epm1_admins = "data/00-raw-symlinks/epm1_administrations.tsv",
		epm1_prescr_header = "data/00-raw-symlinks/epm1_prescr_header.tsv",
		epm1_prescriptions = "data/00-raw-symlinks/epm1_prescriptions.tsv",
		epm1_mapping = "data/00-raw-symlinks/epm1_mapping_prescr_admins.tsv",
		opus_header = "data/00-raw-symlinks/opus_header.tsv",
		opus_admins = "data/00-raw-symlinks/opus_administrations.tsv",
		adms_with_low_egfr = "data/01-admissions_with_low_egfr.tsv"
	output:
		all_drug_admins = "data/01-drug_administrations.tsv",
		target_drug_admins = "data/01-target_drug_administrations.tsv"
	params: target_atc = expand("{atc}", atc = TARGET_DRUGS),
	threads: 5 
	benchmark: "benchmarks/01-fetch_drug_administrations.tsv"
	resources: vmem = 1024*100, tmin = 60*2
	script: "scripts/01-fetch_drug_administrations.R"
	
rule create_cohort: 
	input:
		adms_with_low_egfr = "data/01-admissions_with_low_egfr.tsv",
		target_drug_admins = "data/01-target_drug_administrations.tsv",
		eligible_admissions = "data/01-eligible_admissions.tsv"
	output: tstamp = "data/01-create_cohort.tstamp"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		target_atc = expand("{atc}", atc = TARGET_DRUGS)
	benchmark: "benchmarks/01-create_cohort.tsv"
	resources: vmem = 1024*5, tmin = 60
	script: "scripts/01-create_cohort.R"

rule derive_egfr_eras: 
	input: 
		egfr = "data/01-egfr.tsv",
		cohort = "data/01-create_cohort.tstamp"
	output: egfr_eras = "data/01-egfr_eras.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		fu_time = 30, # days
	benchmark: "benchmarks/derive_egfr_eras.tsv"
	resources: vmem = 1024*5, tmin = 60
	script: "scripts/01-derive_egfr_eras.R"

rule derive_outcome_variables: 
	input:
		egfr_eras = "data/01-egfr_eras.tsv",
		target_drug_admins = "data/01-target_drug_administrations.tsv",
		cohort = "data/01-create_cohort.tstamp"
	output: outcome_variables = "data/01-outcome_variables.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		fu_time = 30, # days
	benchmark: "benchmarks/01-derive_outcome_variables.tsv"
	resources: vmem = 1024*5, tmin = 60
	script: "scripts/01-derive_outcome_variables.R"

# ========== 02 FEATURE ENGINEERING ==========
rule n_previous_admissions: 
	input: "data/01-create_cohort.tstamp"
	output: "data/02-n_previous_admissions.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		lookback = 5, # years
	benchmark: "benchmarks/02-n_previous_admissions.tsv",
	resources: vmem = 1024*5, tmin = 30
	script: "scripts/02-n_previous_admissions.R"

rule atc_level2_given_before_index:
	input:
		all_drug_admins = "data/01-drug_administrations.tsv",
		cohort = "data/01-create_cohort.tstamp"
	output: fpath = "data/02-atc_level2_given_before_index.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		min_patient_per_atc = 10,
	benchmark: "benchmarks/02-atc_level2_given_before_index.tsv",
	resources: vmem = 1024*15, tmin = 60
	script: "scripts/02-atc_level2_given_before_index.R"
	
rule n_distinct_drugs_given_before_index:
	input: 
		all_drug_admins = "data/01-drug_administrations.tsv",
		cohort = "data/01-create_cohort.tstamp"
	output: fpath = "data/02-n_distinct_drugs_given_before_index.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort"
	benchmark: "benchmarks/02-n_distinct_drugs_given_before_index.tsv"
	resources: vmem = 1024*50, tmin = 60
	script: "scripts/02-n_distinct_drugs_given_before_index.R"

rule icd10_chapters_and_blocks_past_5_years:
	input:
		icd10_classification = "data/00-raw-symlinks/icd10_classification.tsv",
		cohort = "data/01-create_cohort.tstamp"
	output: 
		icd10_chapters = "data/02-icd10_chapters_past_5_years.tsv",
		icd10_blocks = "data/02-icd10_blocks_past_5_years.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		lpr_schema = "lpr_dev"
	benchmark: "benchmarks/02-icd10_chapters_and_blocks_past_5_years.tsv"
	resources: vmem = 1024*5, tmin = 60*2
	script: "scripts/02-icd10_chapters_and_blocks_past_5_years.R"

rule elixhauser_scores: 
	input: "data/01-create_cohort.tstamp"
	output: elixhauser_scores = "data/02-elixhauser_scores.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort", lpr_schema = "lpr_dev"
	benchmark: "benchmarks/02-elixhauser_scores.tsv"
	resources: vmem = 1024*5, tmin = 60
	script: "scripts/02-elixhauser_scores.R"
	
rule n18_dx_past_5_years:
	input: "data/01-create_cohort.tstamp"
	output: n18_dx = "data/02-n18_dx_past_5_years.tsv"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort",
		lpr_schema = "lpr_dev"
	benchmark: "benchmarks/02-n18_dx_past_5_years.tsv"
	resources: vmem = 1024*5, tmin = 60
	script: "scripts/02-n18_dx_past_5_years.R"

# ========== 03 DATA SETS ==========
rule create_datasets:
	input:
		outcome_variables = "data/01-outcome_variables.tsv",
		n_previous_admissions = "data/02-n_previous_admissions.tsv",
		atc_level2_given_before_index = "data/02-atc_level2_given_before_index.tsv",
		n_distinct_drugs_given_before_index = "data/02-n_distinct_drugs_given_before_index.tsv",
		icd10_chapters_past_5_years = "data/02-icd10_chapters_past_5_years.tsv",
		icd10_blocks_past_5_years = "data/02-icd10_blocks_past_5_years.tsv", # not currently used
		elixhauser_scores = "data/02-elixhauser_scores.tsv",
		n18_dx_past_5_years = "data/02-n18_dx_past_5_years.tsv"
	output: tstamp = "data/03-create_datasets.tstamp"
	params: dbname = "bth", dbuser = "benkaa", cohort_schema = "hidosfaikid", cohort_table = "cohort"
	benchmark: "benchmarks/03-create_datasets.tsv"
	resources: vmem = 1024*5, tmin = 60*2
	script: "scripts/03-create_datasets.R"
		
# ========== 04 TRAIN MODELS ==========
rule prepare_database_for_optuna:
	input: "data/03-create_datasets.tstamp",
	output: tstamp = "data/04-prep_database_for_optuna.tstamp",
	params: dbname = "bth", dbuser = "benkaa", dbschema = "hidosfaikid"
	benchmark: "benchmarks/04-prepare_database_for_optuna.tsv"
	resources: vmem = 1024, tmin = 10
	script: "scripts/04-prep_database_for_optuna.py"

rule create_directories:
	input: "data/03-create_datasets.tstamp"
	output: "data/04-create_directories.tstamp"
	benchmark: "benchmarks/04-create_directories.tsv"
	resources: vmem = 1024, tmin = 10
	shell: """
		for dir in 04-model_configs 04-weights 04-optuna 04-final 05-optuna_learning 05-performance 05-shap 05-decision_curves
		do
			rm data/$dir -r
			mkdir data/$dir
			printf "*.pickle\n*.hdf5\n*.png\n*.ggplot" > data/$dir/.gitignore
		done
		date > {output}
	"""

rule create_model_configs:
	input: "data/04-create_directories.tstamp",
	output: expand("data/04-model_configs/{mc}.config", mc=MODEL_CONFIGS)
	benchmark: "benchmarks/04-create_model_configs.tsv"
	resources: vmem = 1024, tmin = 10
	shell: "for f in {output}; do date > $f; done"
	
rule optimise_hyperparameters: 
	input: 
		"data/04-model_configs/{outcome_type}__{model_type}__{outcome_variable}.config",
		"data/04-prep_database_for_optuna.tstamp"
	output: 
		fpath_pickle = "data/04-optuna/study__{outcome_type}__{model_type}__{outcome_variable}.pickle",
		fpath_trials_df = "data/04-optuna/study_trials__{outcome_type}__{model_type}__{outcome_variable}.tsv"
	params:
		weights_dir = "data/04-weights",
		n_optuna_trials = 100,
		n_epochs = 400,
		early_stop_patience = 50,
		n_cv_folds = 5,
		dbname = "benkaa",
		dbuser = "benkaa",
		dbschema = "hidosfaikid"
	log: "logs/04-optimise_hyperparameters__{outcome_type}__{model_type}__{outcome_variable}.log"
	benchmark: "benchmarks/04-optimise_hyperparameters__{outcome_type}__{model_type}__{outcome_variable}.tsv"
	resources: vmem = 1024*100, tmin = 60*12
	threads: 5
	script: "scripts/04-run_optuna_studies.py"
	
rule train_explain_final_models: 
	input: rules.optimise_hyperparameters.output
	output: "data/04-final/hist_final__{outcome_type}__{model_type}__{outcome_variable}.tsv"
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid", weights_dir = "data/04-weights"
	log: "logs/04-train_final_models__{outcome_type}__{model_type}__{outcome_variable}.log"
	benchmark: "benchmarks/04-train_final_models__{outcome_type}__{model_type}__{outcome_variable}.tsv"
	resources: vmem = 1024*10, tmin = 60
	script: "scripts/04-train_explain_final_models.py"

# ========== 05 OUTPUTS ==========
# figure 1: drawn by hand, not "data-driven"

rule table_1_summary_stats:
	input: "data/03-create_datasets.tstamp", outcome_variables = "data/01-outcome_variables.tsv"
	output: fpath = "data/05-table_1_summary_stats.tsv"
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid"
	log: "logs/05-table_1_summary_stats.log"
	benchmark: "benchmarks/05-table_1_summary_stats.tsv"
	resources: vmem = 1024*2, tmin = 15
	script: "scripts/05-table_1_summary_stats.R"

rule figure2_bivariate_shap_figure:
	input: expand("data/04-final/hist_final__{x}.tsv", x=MODEL_CONFIGS), rules.create_datasets.output
	output: 
		plot_fname = "data/05-bivariate_shap_plot.pdf",
		plot_ggbuild = "data/05-bivariate_shap_plot.ggbuild"
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid", unit = "cm", height = 25, width = 29,
		features = BIVARIATE_SHAP_FEATURES
	resources: vmem = 1025*5, tmin = 30
	log: "logs/05-bivariate_shap_figure.log"
	script: "scripts/05-bivariate_shap_figure.R"

rule visualise_optuna_learning:
	input: "data/04-optuna/study__{outcome_type}__{model_type}__{outcome_variable}.pickle",
	output: fpath = "data/05-optuna_learning/{outcome_type}__{model_type}__{outcome_variable}.pdf"
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid", unit = "cm", height = 18, width = 28
	resources: vmem = 1024*2, tmin = 15
	script: "scripts/05-optuna_learning.R"
	
rule performance_figures:
	input: "data/04-final/hist_final__{outcome_type}__{model_type}__{outcome_variable}.tsv"
	output:
		calibration_plots = "data/05-performance/calibration_plots__{outcome_type}__{model_type}__{outcome_variable}.pdf",
		roc_curves = "data/05-performance/roc_curves__{outcome_type}__{model_type}__{outcome_variable}.pdf",
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid", unit = "cm", height = 10, width = 28
	resources: vmem = 1024, tmin = 5
	script: "scripts/05-performance_figures.R"
	
rule shap_figure:
	input: 
		"data/04-final/hist_final__{outcome_type}__{model_type}__{outcome_variable}.tsv",
		atc_map = "data/00-raw-symlinks/atc_classification.tsv",
		icd10_classification = "data/00-raw-symlinks/icd10_classification.tsv"
	output: fpath = "data/05-shap/{outcome_type}__{model_type}__{outcome_variable}.pdf"
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid", unit = "cm", height = 28, width = 18
	resources: vmem = 1024, tmin = 5
	script: "scripts/05-shap_figure.R" 
	
rule decision_curve:
	input: "data/04-final/hist_final__{outcome_type}__{model_type}__{outcome_variable}.tsv",
	output: fpath = "data/05-decision_curves/{outcome_type}__{model_type}__{outcome_variable}.pdf"
	params: dbname = "benkaa", dbuser = "benkaa", dbschema = "hidosfaikid", unit = "cm", height = 18, width = 28
	resources: vmem = 1024, tmin = 5
	script: "scripts/05-decision_curve.R" 

rule rulegraph:
	output: 
		dag = "data/05-snakemake_dag.pdf", 
		rg_pdf = "data/05-snakemake_rulegraph.pdf", 
		rg_png = "data/05-snakemake_rulegraph.png"
	resources: vmem = 1024, tmin = 10
	shell: """
		snakemake --dag | dot -Tpdf > {output.dag}
		snakemake --rulegraph | dot -Tpdf > {output.rg_pdf}
		snakemake --rulegraph | dot -Tpng > {output.rg_png}
	"""

rule supplement:
	input:
		rules.table_1_summary_stats.output,
		rules.rulegraph.output,
		expand("data/05-optuna_learning/{x}.pdf", x=MODEL_CONFIGS),
		expand("data/05-performance/calibration_plots__{x}.pdf", x=MODEL_CONFIGS),
		expand("data/05-shap/{x}.pdf", x=MODEL_CONFIGS),
		expand("data/05-decision_curves/{x}.pdf", x=MODEL_CONFIGS),
		rmd_fpath = "data/06-supplement_to_pdf.Rmd"
	output: "data/06-supplement.pdf"
	resources: vmem = 1024, tmin = 10
	shell: """
		module load texlive/2021 # make pdflatex available
		Rscript -e "rmarkdown::render(input = '{input.rmd_fpath}', output_file = '{output}', output_dir = 'data')"
		rm data/core.* # clean-up, possibly needed because ghostscript exits with errors (for some reason)  
	"""
