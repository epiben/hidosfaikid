source("scripts/utils.R")

status("Downloading from database")
conn <- connect()
q <- "
SELECT admission_id, diag_code AS n18_diag
FROM @cohort_schema.@cohort_table AS cohort
INNER JOIN @lpr_schema.adm
	ON adm.person_id = cohort.person_id
	AND adm.departure_date BETWEEN admission_datetime::date - INTERVAL '5 years' AND admission_datetime::date
INNER JOIN @lpr_schema.diag
	ON diag.visit_id = adm.visit_id
	AND diag.diag_code LIKE 'DN18%'
"
n18_dx <- sql_fetch(q, conn,
					cohort_schema = snakemake@params$cohort_schema,
					cohort_table = snakemake@params$cohort_table,
					lpr_schema = snakemake@params$lpr_schema)
dbDisconnect(conn)

status("Wrangling")
n18_dx[, n18_diag := str_sub(n18_diag, 2, 5)]
n18_dx[, n18_diag := if_else(n18_diag %in% paste0("N18", 1:5), n18_diag, "N189")]
n18_dx <- unique(n18_dx)

status("Writing output")
fwrite(n18_dx, snakemake@output$n18_dx, row.names = FALSE, sep = "\t")

status("Done")
