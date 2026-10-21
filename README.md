# Bookstore analytics and BI dashboard

A bookstore analytics service with a Streamlit dashboard, FastAPI endpoints, multi-currency data cleaning, customer identity reconciliation, and a Gemini-powered data copilot.

## Project structure

```text
Task4_PY/
├── data/                           # Store source files
├── src/
│   └── bookstore_bi/
│       ├── assets/
│       │   └── styles.css          # Dashboard styles
│       ├── analytics.py            # Store access, filters, query tools
│       ├── api.py                  # FastAPI application and routes
│       ├── config.py               # Environment and data directory
│       ├── dashboard.py            # Streamlit dashboard and copilot UI
│       ├── llm.py                  # Gemini client and assistant behavior
│       └── pipeline.py             # Loading, cleaning, reconciliation, metrics
├── tests/
│   └── test_analytics.py
├── .env.example
├── pyproject.toml                  # Package and dependency metadata
└── requirements.txt                # Editable install with development extras
```

The project includes source files, a processing pipeline, analytics, and dashboard and API interfaces. The pipeline and analytics modules do not depend on Streamlit, so their logic can be reused by either interface.

## Quickstart

### Install

Use a virtual environment, then install the project and its development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` installs the package in editable mode, with dependency versions and package configuration maintained in `pyproject.toml`.

### Configure

Copy `.env.example` to `.env` and provide a Gemini API key:

```env
GEMINI_API_KEY=replace_with_gemini_api_key
GEMINI_MODEL=gemini-3.7-flash
```

`DATA_DIR` is optional. Relative paths are resolved from the project root; by default, the app reads `data/`.

### Run the Streamlit dashboard

```powershell
streamlit run src/bookstore_bi/dashboard.py
```

### Run the FastAPI service

```powershell
uvicorn bookstore_bi.api:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

### Run tests

```powershell
pytest
```

## Data cleaning and business logic

- Prices are normalized to USD. EUR conversion uses `1 EUR = 1.20 USD` with decimal cents rounding.
- Supported timestamps include common US and European formats, ISO layouts, and 12/24-hour times.
- Known null markers are normalized during cleaning.
- Customer aliases are reconciled using matching identity fields and connected components.
- `author_set` treats a credited co-author combination as a collective unit; `author` credits each co-author individually.
