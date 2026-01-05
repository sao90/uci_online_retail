# Architecture

## Repository Structure

This project follows MLOps best practices with clear separation between reusable business logic and deployment artifacts. The structure is designed to support both local development and cloud deployment.

>>NB: All cloud services are mocked as local counterparts, such as the data layer being represented as a local SQLite database.


### System's Architectural Layers

The application follows a **layered architecture** with clear separation between stateless application logic and external systems:

```
    Stateless Application Layers               Statefull Dependencies

┌───────────────────────────────────┐
│  Layer 1: Entry Point             │
│  Role: CLI routing & environment  │
│  Dir: src/__main__.py             │
└───────────────────────────────────┘
              ↓
┌───────────────────────────────────┐         ┌──────────────────────┐
│  Layer 2: Orchestration           │────────→│  Configuration       │
│  Role: Execute components in      │  reads  │  Dir: app_config/    │
│        sequence, manage data flow,│         │  *.yaml files        │
│        deployment (TODO)          │         │                      │
│  Dir: src/pipelines/              │         └──────────────────────┘
│       local_runner.py             │
│       azure_deployment.py         │
│       (placeholder)               │
└───────────────────────────────────┘
              ↓
┌───────────────────────────────────┐         ┌──────────────────────┐
│  Layer 3: Components              │────────→│  External Systems    │
│  Role: Artifact I/O & CLI         │ read/   │  • Database (SQLite) │
│  Dir: src/components/             │ write   │  • Files (parquet)   │
└───────────────────────────────────┘         │  • Models (pickle)   │
              ↓                               │  • Future: Cloud     │
┌───────────────────────────────────┐         └──────────────────────┘
│  Layer 4: Business & ML Logic     │
│  Role: Isolating data/ML          │
│        functionality              │
│  Dir: src/modules/                │
└───────────────────────────────────┘
```

**Key Principles:**
- **Stateless core**: Layers 1-4 contain no state, only transformation logic
- **External state**: Configuration and data live outside the application source code
- **Unidirectional flow**: Each layer depends only on layers below it
- **Reusability**: Business logic (Layer 4) works in any context (ideally, components (layer 3) should be reusable, but takes longer to design well)


### Pipelines and their components
Pipelines consist of one or more components that perform a specific task. Each component is a self-contained python script (see `src/components/<pipeline>/`), that can be configured with arguments in `app_config/*_pipeline.yaml`.

The three pipelines in the MLOps project are: Preprocessing, Training and Evaluation.
Each pipeline can be run independently or in sequence (TODO: only locally so far).

The pipelines are structured as follows:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│    Component 1:        Component 2:        Component 3:      Component 4:   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐   │
│  │ ingest_data  │───→│  clean_data  │───→│feature_      │───→│split_data│   │
│  │              │    │              │    │ engineering  │    │          │   │
│  │              │    │              │    │              │    │          │   │
│  │ DB → parquet │    │ Transforming │    │ Create       │    │Time-based│   │
│  │              │    │ & validation │    │ covariates   │    │train/test│   │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘   │
│         ↓                   ↓                    ↓                  ↓       │
│   raw_data.parquet   cleaned_data.parquet  features.parquet  train/test     │
│                                                               splits.parquet│
│                                                                             │
│  Purpose: - Configure data needed for model training/evaluation             │
│           - Produce the output                                              │
│           - Isolated to be reused by inference pipeline/service daily.      │
│  Output:  - Model-ready data files (2 covariate files, 2 target var files)  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│           Component 1:                        Component 2:                  │
│        ┌──────────────────┐              ┌─────────────────────┐            │
│        │  train_model     │─────────────→│  backtest_model     │            │
│        │                  │  model.pkl   │                     │            │
│        │ • Loads train    │              │ • Loads model.pkl   │            │
│        │   data           │              │ • Loads train data  │            │
│        │ • Trains model   │              │ • Backtests on      │            │
│        │ • Saves model    │              │   training set      │            │
│        │                  │              │   (start=0.7)       │            │
│        └──────────────────┘              │ • Saves scores.json │            │
│                 ↓                        └─────────────────────┘            │
│                                                   ↓                         │
│             model.pkl                         scores.json                   │
│                                                                             │
│  Purpose: - Train models and validate performance on training data          │
│           - Iterate manually to find good model hyperparameters             │
│  Output:  - 1 model.pkl + scores.json                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Component 1:                                        │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    backtest_model                            │           │
│  │                                                              │           │
│  │  • Loads multiple models (model1.pkl, model2.pkl, ...)       │           │
│  │  • Loads train + test data                                   │           │
│  │  • Concatenates: train + test                                │           │
│  │  • Backtests each model on full data:                        │           │
│  │    - backtest start=split_date (test boundary)               │           │
│  │    - retrain=True simulates (assumed) production requirement │           │
│  │  • Saves evaluation_results.json                             │           │
│  │                                                              │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                              ↓                                              │
│                  evaluation_results.json                                    │
│                  {                                                          │
│                    "model1": {"rmse": 123, "wmape": 12.3},                  │
│                    "model2": {"rmse": 456, "wmape": 45.6}                   │
│                  }                                                          │
│                                                                             │
│  Purpose: - Compare multiple models on unseen test data.                    │
│           - Find best performer among exisiting models for inference.       │
│           - DO NOT(!) run with intend of improving model (bias/leakage).    │
│           - Run periodically, or trigger with inference drift               │
│  Output:  - Scores for multiple models → Select champion                    │
└─────────────────────────────────────────────────────────────────────────────┘

```

**Key Design Decisions**:
- **Preprocessing pipeline:** seperated from training/evaluation. must be run independently before any of the others to prepare desired data (assume production, where datapoints expand daily)
- **Training pipeline:** Backtest on training data (iterate manually to improve model here)
- **Evaluation pipeline:** Backtest multiple champion candidates on train+test (start=test_boundary) to select champion for inference
- **ML Artifacts** are passed between components are stored locally (would be MLFlow and cloud-based in production)
- **Avoiding data leakage:** Covariates always provided as full dataset - Darts library handles internal slicing to avoid leakage

### Directory Layout

```
uci_online_retail/
├── src/                              # All application Python code
│   ├── __main__.py                  # Main entry point (python -m src)
│   │
│   ├── modules/                     # Reusable modules
│   │   ├── data_processing/
│   │   │   ├── data_loader.py       # SQL → DataFrame
│   │   │   ├── data_cleaner.py      # Data cleaning transformations
│   │   │   ├── data_splitter.py     # Time-based data splitting
│   │   │   └── feature_engineer.py  # Feature engineering logic
│   │   ├── model_handling/
│   │   │   └── model_catalogue.py   # Model configurations
│   │   ├── log_config.py            # Logging setup
│   │   └── utils.py                 # Shared utilities
│   │
│   ├── components/                  # Pipeline components (thin wrappers)
│   │   ├── preprocessing/
│   │   │   ├── ingest_data.py       # CLI wrapper for DataLoader
│   │   │   ├── clean_data.py        # CLI wrapper for DataCleaner
│   │   │   ├── split_data.py        # CLI wrapper for DataSplitter
│   │   │   └── feature_engineering.py  # CLI wrapper for FeatureEngineer
│   │   ├── training/
│   │   │   └── train_model.py       # Model training component
│   │   └── evaluation/
│   │
│   ├── pipelines/                   # Pipeline orchestration
│   │   ├── local_runner.py          # Local pipeline execution script
│   │   └── Azule_deployment.py      # Orchestrating cloud deployment (placeholder)
│   │
│   └── setup_scripts/               # Database initialization & utilities
│       └── initialize_sqlite_database.py
│
├── app_config/                       # Pipeline configuration files (Azure ML style)
│   ├── dev/                         # Development environment configs
│   │   ├── preprocessing_pipeline.yaml
│   │   ├── training_pipeline.yaml
│   │   └── evaluation_pipeline.yaml
│   ├── test/                        # Test environment configs
│   └── prod/                        # Production environment configs
│
├── data/
│   ├── retail.db                    # Local SQLite database
│   └── pipeline_runs/               # Local pipeline artifacts (gitignored)
│
└── tests/                            # Test suite (placeholder)
    ├── unit_tests/
    └── integration/
```

---

## Layer 1: Entry Point

### Purpose & Responsibility
- Single CLI interface for all pipelines
- Route to local or cloud execution
- Environment selection (dev/test/prod)

### Usage

```bash
# Run single pipeline locally (dev environment by default)
python -m src --pipelines preprocessing --run_locally True

# Run multiple pipelines
python -m src --pipelines preprocessing training --run_locally True

# Specify environment
python -m src --pipelines preprocessing --run_locally True --environment prod
```

**Arguments:**
- `--pipelines`: One or more pipelines (choices: preprocessing, training, evaluation)
- `--run_locally`: Run locally or deploy (choices: True, true, False, false; default: True)
- `--environment`: Target environment (choices: dev, test, prod; default: dev)

---

## Layer 2: Pipeline Orchestration

### Purpose & Responsibility
- Read YAML configuration files
- Execute components in sequence
- Manage inter-component data flow and dependencies

### Usage

**Local execution** (`local_runner.py`):
- Invoked automatically by Layer 1
- Reads config from `app_config/{environment}/*.yaml`
- Executes components as subprocesses
- Simulates Azure ML's process isolation

**Cloud deployment** (`azure_deployment.py`):
- Placeholder for future Azure ML integration
- Will submit YAML pipelines to cloud

**Configuration** (YAML files):
- Located in `app_config/{dev,test,prod}/`
- Define component paths, inputs, outputs, dependencies

---

## Layer 3: Components

### Purpose & Responsibility
- Thin CLI wrappers around business logic modules
- Handle file I/O (read/write parquet, pickle, database)
- Parse command-line arguments
- Manage logging and error handling

### Usage

**Via pipeline** (recommended):
```bash
python -m src --pipelines preprocessing --run_locally True
```

**Direct execution** (for testing individual components):
```bash
python -m src.components.preprocessing.ingest_data \
    --db_path data/retail.db \
    --table_name online_retail \
    --output_data data/pipeline_runs/raw_data.parquet
```

**Component structure:**
```python
def parse_args():
    """Parse CLI arguments."""
    # argparse setup

def main():
    """Handle I/O and orchestration."""
    args = parse_args()
    # Use business logic from Layer 4
    # Handle file I/O
```

---

## Layer 4: Business Logic

### Purpose & Responsibility
- Pure transformation logic (DataFrame → DataFrame, model training, etc.)
- Reusable across training, evaluation, inference services
- No I/O operations, no CLI dependencies
- Easily testable

### Usage

```python
# Import and use in any context
from src.modules.data_processing.data_cleaner import DataCleaner
from src.modules.model_handling.model_catalogue import MODEL_CATALOGUE

# Use in component
cleaner = DataCleaner()
cleaned_df = cleaner.run(raw_df)

# Use in inference service
model_fn = MODEL_CATALOGUE["random_forest_1111"]
model = model_fn()  # Get fresh instance
predictions = model.predict(data)
```

**Module organization:**
- `src/modules/data_processing/` - Data transformation classes
- `src/modules/model_handling/` - Model configurations and factories
- `src/modules/utils.py` - Shared utilities
- `src/modules/log_config.py` - Logging setup

---
