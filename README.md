# UCI online retail MLops

# Quickstart
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

3) Create a virtual environment and install dependencies in your project directory
    ```bash
    pdm install -G:all
    ```
4) Activate virtual environment
    ```bash
    pdm venv activate
    ```
    and run the command displayed in your terminal (e.g., `source .venv/bin/activate`).

5) Setup pre-commit hooks
   ```bash
   pre-commit install --install-hooks
   ```


### 2. Setup environment variables
`.env-example` file contains example environment variables cleaned for secrets. Copy it, change local paths and name the new file `.env`

### 3. Setup mocked database backend

`src/setup_scripts/initialize_sqlite_database.py` will mock a database backend based on the excel file provided by UCI. table and database names are defined in .env file.


## Running the application

### Quick Start

To run the ML pipelines, use this command:

```bash
python -m src --pipelines <pipeline_name> --run_locally True --environment <env>
```

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



## Case text
```txt
I denne case skal du analysere UCI Online Retail* datasættet. Du må selv vælge hvad du
vælger at undersøge, men du kan finde inspiration i følgende liste:
- Kundesegmentering
- Kurv-analyse & produktanbefalinger
- Salgsprognoser

Som en del af analysen ser vi gerne at du kommer ind på følgende:

- Eksplorativ dataanalyse og simple datarensning - Hvilke udfordringer ser du i dataen
og hvad kan man gøre ved det?

- Feature Engineering – Skab 2-3 relevante features.

- Model-/metodevalg og træning – Vælg og træn en model/metode til din
problemstilling.

- Eksperimentdesign og validering – Beskriv hvordan du vil designe dit eksperiment,
bl.a.:
  - Hvilke evalueringsmetrikker vil du vælge?
  -  Hvordan sikre du at dit resultat generaliserer?
  - Bonus: Hvad skal der til for at kunne sætte din løsning i produktion?

Vi lægger vægt på dine overvejelser og din problemløsning frem for modelperformance og
resultater. Vi forventer ikke en fuldt implementeret løsning, men vil se, at du demonstrerer
dine evner til at skrive god Python-kode

UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail

```
