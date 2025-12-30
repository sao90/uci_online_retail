# Architecture

## Repository Structure

This project follows MLOps best practices with clear separation between reusable business logic and deployment artifacts. The structure is designed to support both local development and cloud deployment.

>>NB: All cloud services are mocked as local counterparts, such as the data layer being represented as a local SQLite database. No terraform ressources have been defined in this repository either.

### Directory Layout

```
uci_online_retail/
├── src/                              # Reusable business logic
│   └── data_processing/
│       ├── data_loader.py           # SQL → DataFrame
│       ├── data_cleaner.py          # Data cleaning transformations
│       ├── data_splitter.py         # Time-based data splitting
│       └── feature_engineer.py      # Feature engineering logic
│
├── components/                       # Pipeline components (thin wrappers)
│   ├── preprocessing/
│   │   ├── ingest_data.py          # CLI wrapper for DataLoader
│   │   ├── clean_data.py           # CLI wrapper for DataCleaner
│   │   ├── split_data.py           # CLI wrapper for DataSplitter
│   │   └── feature_engineering.py  # CLI wrapper for FeatureEngineer
│   ├── training/
│   └── evaluation/
│
├── pipelines/                        # Pipeline definitions & orchestration
│   ├── local_runner.py              # Local pipeline execution script
│   ├── preprocessing_pipeline.yaml  # Azure ML pipeline definition
│   ├── training_pipeline.yaml
│   └── evaluation_pipeline.yaml
│
├── data/
│   ├── retail.db                    # Local SQLite database
│   └── pipeline_runs/               # Local pipeline artifacts (gitignored)
│
└── notebooks/                        # Exploratory analysis
```

### Architectural Principles

#### 1. Separation of Concerns

**Core Business Logic (`src/`):**
- Stateless, reusable Python classes
- Pure transformations: DataFrame → DataFrame
- No CLI arguments, no file I/O
- Easily unit testable
- Used by both training layer and serving layer

**Pipeline Components (`components/`):**
- Thin CLI wrappers around `src/` classes
- Handle argument parsing (argparse)
- Manage file I/O (read/write files)
- Orchestration and logging

**Pipeline Definitions (`pipelines/`):**
- YAML files define Azure ML pipeline orchestration
- Local runner script for development testing
- No business logic - pure orchestration

#### 2. Component Design Pattern

Each component follows this structure:

```python
# components/preprocessing/ingest_data.py

def parse_args():
    """Parse CLI arguments (Azure ML compliant)."""
    # --input_data, --output_data, etc.

def main():
    """Entry point - handles I/O and orchestration."""
    args = parse_args()

    # Use shared business logic
    loader = DataLoader(db_path=args.db_path)
    df = loader.load_table(table_name=args.table_name)

    # Handle file I/O (component responsibility)
    df.to_parquet(args.output_data, index=False)

if __name__ == "__main__":
    main()
```

**Benefits:**
- Business logic is reusable across pipelines and inference
- Components remain Azure ML compatible (simple Python scripts)
- Easy to test components independently
- Clear boundaries between computation and I/O

#### 3. Execution Models

**Local Development:**
```bash
# Run full pipeline locally
python -m pipelines.local_runner

# Run single component for debugging
python components/preprocessing/ingest_data.py \
  --db_path data/retail.db \
  --output_data data/pipeline_runs/raw_data.parquet
```

The local runner (`pipelines/local_runner.py`) simulates Azure ML by:
- Executing components as separate subprocess calls
- Passing file paths as CLI arguments
- Using local filesystem as mock blob storage (`data/pipeline_runs/`)

**Azure ML Deployment:**
```bash
# Submit pipeline to Azure ML
az ml job create --file pipelines/preprocessing_pipeline.yaml
```

Azure ML:
- Reads the YAML pipeline definition
- Provisions compute resources
- Executes each component as isolated process
- Manages data passing via blob storage URIs

#### 4. Data Flow

**Preprocessing Pipeline:**
```
SQL Database
    ↓
[ingest_data] → raw_data.parquet
    ↓
[clean_data] → cleaned_data.parquet
    ↓
[split_data] → train.parquet, val.parquet, test.parquet
    ↓
[feature_engineering] → train_featured.parquet, val_featured.parquet, test_featured.parquet
```

Each step:
1. Reads input from previous step (parquet file)
2. Imports and instantiates class from `src/data_processing/`
3. Calls transformation methods
4. Writes output (parquet file) for next step

#### 5. Reusability Across Contexts

The same `src/` classes can be used across pipelines and services. For instance, a prediction class will be used both in training pipeline, evaluation pipeline, and a inference service. It's core logic should be isolated to the 1 class to avoid inconsistencies in respectively the training- and serving layer of the system.


### Entry Points

**For Local Development:**
- `pipelines/local_runner.py` - Execute full pipeline locally for faster development iterations. Ensuring consistent arguments passed for local and cloud-based executions is not yet refined. It should be handled better than here.

**For Azure ML:**
- `pipelines/preprocessing_pipeline.yaml` - Submit to Azure ML for cloud execution

**For Individual Components:**
- `components/preprocessing/ingest_data.py` - Can be run standalone for debugging
- Each component is a valid Python script with CLI interface

### Configuration

- `.env` - Environment variables (database paths, table names, local directories)
- `pyproject.toml` - Python dependencies managed by PDM
- `.vscode/settings.json` - IDE configuration (Python interpreter, analysis paths)
- `pipelines/*.yaml` - Azure ML pipeline definitions (compute, environment, data flow)

### Key Design Decisions

1. **YAML-first for Azure ML** - Pipeline definitions are YAML files, not Python code. This follows Azure ML conventions and makes pipelines declarative and version-controllable.

2. **Components are thin** - Minimal logic in components. All transformation logic lives in `src/` for reusability and testability. Components are lightweight interfaces that handles the IO operations required for the heavier lifting of the modules in `src/`

3. **In-memory object interfaces** - Shared classes operate on already-instantiated Python objects (DataFrames, Darts TimeSeries, model objects), not I/O artifacts (file paths, database connections, URIs). Components handle I/O and pass instantiated objects to shared logic. This separation makes shared classes context-agnostic and easily testable.

4. **Subprocess execution in local runner** - Simulates Azure ML's process isolation model, catching issues before cloud deployment.

5. **No compilation step** - Unlike GCP Vertex AI / Kubeflow, Azure ML doesn't require compiling Python to YAML. Components are referenced directly in YAML definitions.
