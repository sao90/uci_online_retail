## v0.6.0 (2026-01-06)

### Feat

- **training_pipeline.yaml,-training_pipeline_local_runner.py**: implement backtest component in training pipeline
- **model_performance_investigations.ipynb**: add simple example of shap explanation usage. more work required
- **model_performance_investigation.ipynb**: notebook for local model investigations. (could be implemented in training pipeline)

### Fix

- **model_performance_investigation.ipynb**: run pre-commit hooks on notebook

### Refactor

- **components/training/backtest_model.py**: change argument names for consistency + add logging + ad TODO placeholders
- **preprocessing_pipeline.yaml,-__main__.py,-preprocessing_pipeline_runner.py**: refactor preprocessing pipeline execution in local_runner + add comment explaining choice of input references in yaml

## v0.5.0 (2026-01-05)

### Refactor

- **model_handler.py**: avoid using default value. must be passed through config for explicity

## v0.4.0 (2026-01-04)

### Feat

- **backtest_model.py**: add component for backtesting in training pipeline
- **model_handler.py**: added module for model handling (training and backtest)

### Fix

- **investigate_data.ipynb**: backtest experimentation in notebook
- **model_catalogue.py**: fix encoder dict

### Refactor

- **train_model.py**: refactor component to use new ModelHandler module
- **all-python-files**: move individual folders into 'src' and add __main__.py

## v0.3.0 (2026-01-04)

### Feat

- **local_runner,-training_pipeline.yaml**: add model training component to local runner, and create yaml config for training pipeline
- **train_model.py**: model training component using model_catalogue
- **model_catalogue.py**: add model catalogue with small sample of model settings to choose from
- **investigate_data.ipynb**: small notebook for various data and model testing
- **local_runner.py,-utils.py,-preprocessing_pipeline.yaml**: implement AzureML-style config to work with local_runner instead of .env file
- **preprocessing_pipeline.yaml**: move .env variables to azure-compliant pipeline-yaml
- **local_runner.py**: implement feature engineering component to local pipeline runner
- **feature_engineering.py**: add feature engineering component as thin wrapper over FeatureEngineer class
- **feature_engineer.py**: module for performing feature engineering operations on the dataset.

### Fix

- **.gitignore**: add model.pkl from training pipeline output
- **feature_engineer.py**: add buffer for future covariate lags (leads) at model training and prediction time
- **initialize_sqlite_database.py,-data_cleaner.py,-investigate_data.py**: remove debit transactions that were later credited along with the cancelled transactions
- **example.env**: update example env for external users
- **initialize_sqlite_database.py**: add docstring to setup script. Still uses .env file
- **notebook**: update git cache to untrack notebook
- **.gitignore**: add exploratory notebook
- **.gitignore**: add all .parquet files from local pipeline runs
- **cleaned_data.parquet,-raw_data.parquet**: remove local pipeline output files from git tracking

### Refactor

- **data_splitter.py**: keep quantity in features_raw.parquet, for better code in feature_engineering

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
