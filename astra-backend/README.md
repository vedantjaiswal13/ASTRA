# ASTRA Backend (`astra-backend`)

High-performance Python backend service for scientific data ingestion and image processing pipelines, built on FastAPI.

---

## 📁 Project Structure

```text
astra-backend/
├── .venv/                      # Python virtual environment containing dependencies
├── .gitignore                  # Git ignore rules for Python artifacts & venv
├── requirements.txt            # Project dependencies definition
├── README.md                   # Backend documentation and getting started guide
└── app/
    ├── __init__.py             # Marks 'app' as a Python package
    ├── main.py                 # FastAPI application initialization & /health route
    ├── api/
    │   ├── __init__.py         # Aggregator router for modular endpoints
    │   └── routes.py           # Endpoint definitions & controllers
    ├── data/
    │   ├── __init__.py         # Data ingestion package
    │   └── ingestion.py        # S3, NetCDF, xarray data loading services
    └── processing/
        ├── __init__.py         # Processing package
        └── image_processor.py  # Computer vision (OpenCV, skimage) routines
```

---

## 🚀 Getting Started

### 1. Activate the Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

### 3. Run the Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🩺 Health Check Endpoint

- **URL**: `http://localhost:8000/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

Interactive API documentation is accessible at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).
