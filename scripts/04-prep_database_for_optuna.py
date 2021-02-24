from optuna import storages
from sqlalchemy import create_engine, text
from datetime import datetime

"""

This script:
- Removes Optuna tables so results from previous studies do not linger
- Removes old training-related tables (e.g., losses)
- Sets up fresh Optuna tables ready for new studies

"""

training_tables = [
	"studies",
	"version_info",
	"study_user_attributes",
	"study_system_attributes",
	"trials",
	"trial_user_attributes",
	"trial_system_attributes",
	"trial_params",
	"trial_values",
	"alembic_version",
	"predictions",
	"training_summaries",
	"best_trials",
	"evaluation_metrics",
	"shap_values"
]

DBNAME = snakemake.params["dbname"]
DBUSER = snakemake.params["dbuser"]
DBSCHEMA = snakemake.params["dbschema"]
psql_url = f"postgresql://{DBUSER}@dbserver/{DBNAME}?options=-c%20search_path={DBSCHEMA}"

engine = create_engine(psql_url) 
with engine.connect() as connection:
	q = " ".join(f"DROP TABLE IF EXISTS {DBSCHEMA}.{table} CASCADE;" for table in training_tables)
	connection.execute(text("BEGIN; " + q + "COMMIT;")) # remove training-related tables
	storages.RDBStorage(url=psql_url, engine_kwargs={"pool_size": 0}) # initiate empty Optuna tables

with open(snakemake.output[0], "w") as f:
	f.write(str(datetime.now()))
