## v0.2.0 (2025-12-31)

### Feat

- **split_data.py,-data_splitter.py,-local_runner.py**: add module+component for data splitting and integrate in local_runner
- **investigate_data.ipynb**: more investigations
- ***_pipeline.yaml**: placeholder yaml pipeline configurations for Azure-style executions
- **ingest_data-and-clean_data-componenents-in-working-local_runner-pipeline.**: local execution mode set up for the architecture of modules, components and pipelines
- **data_cleaning.py**: add more cleaning methods based on exploration
- **data_cleaning.py**: add query_to_df method
- **investigate_data.ipynb**: placeholder notebook for investigating data sources
- **data_cleaning.py**: first draft of data cleaning class using sql backend for cleaning operations
- **initialize_sqlite_database.py**: script to create local sql database (mocking external DB backend)
- **log_config.py**: configure common logging for modules
- **.env-example**: example .env file for reproducability
- **online_retail_dataset.xlsx,-variable_descriptions.md**: added data in repo

### Fix

- **.gitignore**: add local pipeline output files
- **initialize_dqlite_database.py**: configure DB table name from env file + update example.env
- **initialize_sqlite_database**: move setup_logging() to top of function
- **.gitignore**: ignore .vscode folder for workspace settings
