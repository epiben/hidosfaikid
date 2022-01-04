# HOUSEKEEPING AND UTILITY FUNCTIONS USED IN MULTIPLE PLACES

try(sink(snakemake@log[[1]], split = TRUE, append = TRUE), silent = TRUE) 
try(sink(snakemake@log[[1]], split = TRUE, append = TRUE, type = "message"), silent = TRUE) 

if (!exists("snakemake")) stop("Script must be called via snakemake.", call. = FALSE)

for (p in c("plyr", "tidyverse", "data.table", "tidyr", "lubridate", "RPostgreSQL"))
	library(p, character.only = TRUE)

`%||%` <- function(a, b) if(!is.null(a)) a else b # snatched from tidyverse

# Starting logging to a user-specified file. Returns the log-file path but add a break to the file
start_log <- function(log_file) {
	write_lines("\n===== NEW RUN =====", log_file, append = TRUE)
	return(log_file)
}

# Gives a message with milestone and, optionally, writes to a continuous log file if one such exists in context
status <- function(out) {
	o <- sprintf("%s\t%s", now(), out)
	message(o) # standard out
	cat(o, "\n") # to sink file
}

# DATA WRANGLING FUNCTIONS
concat <- function(x, remove_duplicates = TRUE, sep = ", ") {
	if (isTRUE(remove_duplicates)) x <- unique(x)
	paste(x, collapse = sep)
}

# Save plots as PDF, PNG and ggplot objects
save_plot <- function(p, fname) {
	fname <- str_remove_all(fname, ".pdf")
	ggsave(paste0(fname, ".pdf"), p, units = snakemake@params$unit, width = snakemake@params$width,
		   height = snakemake@params$height)
	ggsave(paste0(fname, ".png"), p, units = snakemake@params$unit, width = snakemake@params$width,
		   height = snakemake@params$height)
	write_rds(p, paste0(fname, ".ggplot"))
}

# Put scaled variable back on original scale
descale <- function(x) { # output from base::scale
	as.numeric(x * attr(x, "scaled:scale") + attr(x, "scaled:center"))
}

# DATABASE FUNCTIONS
connect <- function() {
	dbConnect(dbDriver("PostgreSQL"), host = "trans-db-01", port = 5432, 
			  dbname = snakemake@params$dbname, user = snakemake@params$dbuser)
}

# Run parameterised SQL query on server (from file if such exists)
parse_sql <- function(query, ...) {
	try(query <- read_file(query), silent = TRUE)
	params <- unlist(list(...))
	if (!is.null(params)) query <- str_replace_all(query, setNames(params, paste0("@", names(params))))
	query
}

sql_exec <- function(query, conn, ...) { # run query on server (doesn't fetch anything)
	DBI::dbExecute(conn, parse_sql(query, ...))
}

sql_fetch <- function(query, conn, ...) { # load table from database
	DBI::dbGetQuery(conn, parse_sql(query, ...)) %>% 
		as.data.table()
}

write_to_database <- function(table_name, object, schema = NULL, connection = conn, append = TRUE) {
	if (!is.null(schema)) sql_exec(sprintf("SET SEARCH_PATH TO %s", schema), conn)
	RPostgreSQL::dbWriteTable(connection, table_name, object, append = append, overwrite = !append, row.names = FALSE, 
							  fileEncoding = "UTF-8")
}

load_cohort <- function(cohort_schema = NULL, cohort_table = NULL, conn = conn) {
	cohort_tableschema <- cohort_schema %||% snakemake@params$cohort_schema
	conn <- connect()
	cohort <- sql_fetch("SELECT * FROM @schema.@table_name", conn, 
						schema = cohort_schema %||% snakemake@params$cohort_schema, 
						table_name = cohort_table %||% snakemake@params$cohort_table)
	dbDisconnect(conn)
	return(cohort)
}

icd10_chapters <- c("1" = "Infections",
					"2" = "Neoplasms",
					"3" = "Blood and immune",
					"4" = "Endocrine, nutritional, metabolic",
					"5" = "Psychiatry",
					"6" = "Nervous system",
					"7" = "Eye and adnexa",
					"8" = "Ear and mastoid",
					"9" = "Cardiovascular",
					"10" = "Respiratory",
					"11" = "Digestive system",
					"12" = "Dermatology",
					"13" = "Musculoskelatal and connective tissue",
					"14" = "Genitourinary",
					"15" = "Obstetrics",
					"16" = "Perinatal",
					"17" = "Congenital",
					"18" = "Miscellaenous symptions/findings",
					"19" = "Injury, poisoning, etc.",
					"20" = "External causes",
					"21" = "Other determinants",
					"22" = "Special codes")

# For testing in Rstudio
# setClass("snakemake", slots = c(input = "list", params = "list", output = "list", log = "character", config = "list", thread = "numeric"))
# snakemake <- new("snakemake", params = list(dbname = "bth", dbschema = "lpr_dev", dbuser = "benkaa"))
