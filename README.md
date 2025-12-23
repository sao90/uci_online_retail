# uci_online_retail

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


## Before running the application

### 1. install dependencies

Based on [PDM](https://pdm-project.org/latest/)

1) [Install PDM](https://pdm-project.org/latest/#installation)
    ```bash
    pip install --user pdm
    ```
2) Check the PDM version: currently using version 2.19.3.
    ```bash
    pdm --version
    ```

3) Create a virtual environment in your home repository, including pre-configured "test" and "dev" dependencies.
    ```bash
    pdm install -G:all
    ```

### 2. Setup environment variables
`.env-example` file contains example environment variables cleaned for secrets. Copy it and name the new file `.env`

### 3. Setup mocked database backend

`setup_scripts/initialize_sqlite_database.py` will mock a database backend based on the excel file provided by UCI.
It contains a single table: "transactions" which is a 1:1 copy of the UCI data.
