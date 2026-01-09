# UCI online retail MLops
This repository contains a ML-Ops structure for building and maintaining forecasting models for the UCI online retail dataset.
It is a scaffolding currently working locally, but designed to be compatible with Azure ML pipelines.
It does not contain an inference point for consuming the forecast, but can support deployment of models to inference services or scheduled pipelines writing to a database.

Goals:
1) **Having hands-on operational feel for the models being used in production**
   - How: Data Scientists can define and train models locally with minimal inteference with the Software
            No model is automatically tuned and forwarded to inference. While this can be achieved, this structure believes in
            robust hand-held model optimization and error analyses over trying to build complex software that can 'do-it-all'.
2) **Easily configurable**
   - How: Model maintenance can be achieved by defining a Darts model and running pipelines locally with configurable inputs.
3) **Scalability**
   - How: Designing re-usable components. Currently, the project allows for forecasting any country present in the UCI dataset.
            This principle is easily extended to product-level forecasts
4) **Modular**
   - How: Modules are designed to be re-used for their core functionality. e.g. `src/modules/model_handling/model_handler.py`
            is reusable in any pipeline or service where model operations occur. The preprocessing pipeline is entirely reusable for both training and inference, where data needs processing before model operations can occur.
            Particularly for inference, this is key to ensure that inference code is not duplicated elsewhere, but uses the same operations as the MLops.
            Note: ideally, 1 of the 4 layers in the architecture should be designed away as it is too nested right now. This is doable, but requires more time.
5) **Abstracting complex ML operations away**
   - How: By using Darts, complex time series operations are handled with ample documentation for developers. Building new software features for the pipelines means re-using proven frameworks from others (alternatively, SK Time library offers much of the same). This also outsources a lot of documentation; and developers need only to learn one framework to get started.
6) **Experimentation is easy**
   - How: A common Data Science obstacle to easily experiment in a setting that resembles production. This project isolates
            each pipeline and allows data scientists to run e.g. preprocessing pipeline with given parameters; load the data in a notebook; employ the existing modules or components for defining, initializing, training and benchmarking a model. It saves time and ensures methodology is alligned across experimentation and production.
            For secure experimentation decoupled from production inference, a seperate environement can be set up and embedded in the CI/CD framework (e.g. dev -> test -> experimentation -> prod)



## Before running the application

### 1. Install dependencies

Based on [PDM](https://pdm-project.org/latest/)

1) [Install PDM](https://pdm-project.org/latest/#installation) if not already installed
    ```bash
    pip install --user pdm
    ```
2) Check the PDM version: currently using version 2.19.3.
    ```bash
    pdm --version
    ```

3) **Recommended Python 3.12 is installed**
    This project is build using python 3.12, but may work on other versions too. To be sure, install a python 3.12 version first

   Check if you have Python 3.12:
   ```bash
   python3.12 --version
   ```

   If not installed, install it:
   - **macOS (Homebrew):** `brew install python@3.12`
   - **macOS (pyenv):** `pyenv install 3.12` then `pyenv global 3.12`
   - **Ubuntu/Debian:** `sudo apt install python3.12`
   - **Windows:** Download from [python.org](https://www.python.org/downloads/)

4) Create virtual environment
   ```bash
   pdm venv create 3.12
   ```
   Or if you prefer to select interactively:
   ```bash
   pdm venv create
   ```
   Then select Python 3.12 from the list. **Note:** Only Python 3.12.x versions will appear since this project requires exactly that version (see `pyproject.toml`).

5) Install dependencies (PDM automatically uses the venv created above)
    ```bash
    pdm install -G:all
    ```

6) Setup pre-commit hooks
   ```bash
   pre-commit install --install-hooks
   ```

7) Activate virtual environment (needed for CLI commands)
    ```bash
    eval $(pdm venv activate)
    ```
    Or follow the command displayed in your terminal (e.g., `source .venv/bin/activate`).

    **Note:** Activation is only needed when you want to run `python`, `pytest`, or other commands directly. PDM commands work without activation.


### 2. Setup environment variables
`.env-example` file contains example environment variables cleaned for secrets. Copy it, change local paths and name the new file `.env`

### 3. Setup mocked database backend

`src/setup_scripts/initialize_sqlite_database.py` will mock a database backend based on the excel file provided by UCI. table and database names are defined in .env file.


## Running the application

### Quick Start

To run the ML pipelines, use this command:

```bash
python -m src --args options
```
The command invokes the `__main__.py` script in `src` directory.

### Command Arguments

| Argument | Required | Options | Default | Description |
|----------|----------|---------|---------|-------------|
| `--pipelines` | Yes | `preprocessing`, `training`, `evaluation` | - | One or more pipelines to run (space-separated) |
| `--run_locally` | No | `True`, `true`, `False`, `false` | `True` | Run locally (True) or deploy to cloud (False) |
| `--environment` | No | `dev`, `test`, `prod` | `dev` | Which environment configuration to use |

### Common Examples

**Run preprocessing (development environment):**
```bash
python -m src --pipelines preprocessing --run_locally True
```

**Run preprocessing and training together:**
```bash
python -m src --pipelines preprocessing training --run_locally True
```

**Run with production configuration:**
```bash
python -m src --pipelines training --run_locally True --environment prod
```

### Advanced Usage

**Run individual component for debugging:**
```bash
python -m src.components.preprocessing.ingest_data \
    --db_path data/retail.db \
    --table_name online_retail \
    --output_data data/pipeline_runs/raw_data.parquet
```

For detailed architecture and design documentation, see [ARCHITECTURE.MD](ARCHITECTURE.MD).


## ML ops framework:
