source("scripts/utils.R")

# Setup
setDTthreads(snakemake@threads)
log_file <- start_log("logs/01_fetch_drug_administrations.log")

# Helper function
make_medication_header <- function(file_path) {
	read_file(file_path) %>% 
		tolower() %>% 
		str_split("\t") %>% 
		unlist()
}

# Helper to import and rename columns
header_lookup <- c(cpr = "person_id",
				   pid = "person_id",
				   ord_id = "prescription_id",
				   adm_id = "administration_id",
				   navn = "product_name", 
				   drug_name = "product_name",
				   prim_atc = "atc", 
				   atc_code = "atc",
				   adm_start = "admin_datetime",
				   adm_date = "admin_date",
				   adm_time = "admin_time",
				   date_given = "admin_datetime",
				   adm_dosis = "given_dose", # in EPM you get given_dose right away
				   adm_dosis_enhed = "given_dose_unit", # idem
				   strength = "adm_strength",
				   unit = "adm_strength_unit",
				   number_of_units_given = "adm_volume",
				   dose_unit_code = "adm_volume_unit",
				   vareform = "formulation",
				   form_text = "formulation",
				   adm_status = "admin_status")

final_cols <- c("person_id", "admin_datetime", "given_dose", "given_dose_unit", 
				"product_name", "formulation", "atc", "source")

status("Loading admissions with low_egfr")
admissions_with_low_egfr <- fread(snakemake@input$adms_with_low_egfr)

# EPM3 
status("Loading EPM3")
epm3_header <- make_medication_header(read_file(snakemake@input$epm3_header)) 
epm3_cols <- which(epm3_header %in% names(header_lookup))
epm3_colnames <- paste(header_lookup[epm3_header[epm3_cols]])

epm3_admins <- fread(read_file(snakemake@input$epm3_admins), select = epm3_cols, col.names = epm3_colnames, 
					 sep = "\t", quote = "") %>% 
	.[!grepl("^[TX]", person_id) & admin_status == 7]
epm3_admins[, person_id := parse_number(person_id)] # parse to enable mergin

message("N. patients in EPM3 before filtering:", n_distinct(epm3_admins$person_id))

epm3_admins <- merge(epm3_admins, admissions_with_low_egfr[, "person_id"], by = "person_id")
epm3_admins[, ':='(given_dose_unit = tolower(given_dose_unit),
				   admin_datetime = dmy_hms(admin_datetime),
				   source = "epm3")]
epm3_admins[, setdiff(names(epm3_admins), final_cols) := NULL]

message("N. patients in EPM3 after filtering:", n_distinct(epm3_admins$person_id))

# EPM1
status("Loading EPM1 helper files")
epm1_admins_header <- make_medication_header(read_file(snakemake@input$epm1_admins_header)) 
epm1_admins_cols <- which(epm1_admins_header %in% names(header_lookup))
epm1_admins_colnames <- paste(header_lookup[epm1_admins_header[epm1_admins_cols]])

epm1_prescr_header <- make_medication_header(read_file(snakemake@input$epm1_prescr_header))
epm1_prescr_cols <- which(epm1_prescr_header %in% c("prim_atc", "ord_id"))
epm1_prescr_colnames <- paste(header_lookup[epm1_prescr_header[epm1_prescr_cols]])

epm1_prescr_atc <- fread(read_file(snakemake@input$epm1_prescriptions), select = epm1_prescr_cols, 
						 col.names = epm1_prescr_colnames, sep = "\t", quote = "")

epm1_map <- fread(read_file(snakemake@input$epm1_mapping), select = c("adm_id", "ord_id"),
				  col.names = c("administration_id", "prescription_id"))

status("Loading and wrangling EPM1 administrations")
epm1_admins <- fread(read_file(snakemake@input$epm1_admins), select = epm1_admins_cols, col.names = epm1_admins_colnames, 
					 sep = "\t", quote = "") %>% 
	.[!grepl("^[TX]", person_id) & admin_status == 7]
epm1_admins[, ':='(person_id = parse_number(person_id),
				   admin_datetime = dmy_hms(paste(admin_date, admin_time)),
				   source = "epm1")]

message("N. patients in EPM1 before filtering:", n_distinct(epm1_admins$person_id))

epm1_admins <- merge(epm1_admins, admissions_with_low_egfr[, "person_id"], by = "person_id") %>% 
	merge(epm1_map, by = "administration_id") %>% 
	merge(epm1_prescr_atc, by = "prescription_id") 
epm1_admins[, setdiff(names(epm1_admins), final_cols) := NULL]

message("N. patients in EPM1 after filtering:", n_distinct(epm1_admins$person_id))

# Opusmed
status("Loading Opusmed")
opus_header <- make_medication_header(read_file(snakemake@input$opus_header)) 
opus_cols <- which(opus_header %in% names(header_lookup))
opus_colnames <- paste(header_lookup[opus_header[opus_cols]])

opus_admins <- fread(read_file(snakemake@input$opus_admins), select = opus_cols, col.names = opus_colnames, 
					 na.strings = "NULL") %>% 
	.[-grep("^[TX]", person_id)]

message("N. patients in Opus before filtering:", n_distinct(opus_admins$person_id))

opus_admins[, person_id := parse_number(person_id)]
opus_admins <- merge(opus_admins, admissions_with_low_egfr[, "person_id"], by = "person_id")

message("N. patients in Opus after filtering:", n_distinct(opus_admins$person_id))

opus_admins[, ':='(admin_datetime = ymd_hms(admin_datetime),
				   atc = str_remove_all(atc, ":"),
				   source = "opus",
				   given_dose = adm_strength * adm_volume, 
				   # after scrutinising full data, this is appropriate (so done on the fly), although only needed for target drugs
				   given_dose_unit = if_else(adm_strength_unit == "mg/ml", "mg", tolower(adm_strength_unit)))]

opus_admins[, setdiff(names(opus_admins), final_cols) := NULL]

status("Combining into one table")
all_drug_admins <- rbindlist(list(epm3_admins, epm1_admins, opus_admins), use.names = TRUE) %>% 
	.[!is.na(given_dose) & !is.na(given_dose_unit) & given_dose > 0]

target_drug_admins <- all_drug_admins[grep(sprintf("^(%s)", paste(snakemake@params$target_atc, collapse = "|")), atc)]

status("Writing outputs")
fwrite(all_drug_admins, snakemake@output$all_drug_admins, row.names = FALSE, sep = "\t")
fwrite(target_drug_admins, snakemake@output$target_drug_admins, row.names = FALSE, sep = "\t")
