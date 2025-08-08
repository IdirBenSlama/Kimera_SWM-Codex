# KIMERA SWM - Simple Quickstart (Windows)

This is a personal/experimental project. These steps let you run and check it without deep technical knowledge.

## Prerequisites
- Python 3.11 or newer (add to PATH during install)
- Optional: Docker Desktop (for full database/monitoring stack)

## One-command run
1. Open Windows PowerShell.
2. Navigate to the project folder (this folder).
3. Run:
   ```powershell
   scripts\run_dev.ps1
   ```
   - This creates a virtual environment, installs dependencies, and starts the API in development mode using SQLite.

4. Open the docs at:
   - http://127.0.0.1:8000/docs

## Smoke test (checks that API is up)
```powershell
scripts\smoke_test.ps1
```
- You should see PASS lines for /openapi.json and /docs. (/system/status may pass if available.)

## Full stack (PostgreSQL, Neo4j, Redis, Prometheus, Grafana)
If you have Docker Desktop:
```powershell
scripts\run_dev.ps1 -WithInfra
```
- This will start the databases/monitoring and use PostgreSQL automatically.

## Environment variables (optional)
- `KIMERA_ENV` (default: `development`)
- `DATABASE_URL` (default: SQLite file `kimera_swm.db` unless `-WithInfra`)

## Where is the API?
- Uvicorn target: `src.api.app_main:app`
- Dev server: http://127.0.0.1:8000

## Running tests (optional)
```powershell
.\.venv\Scripts\Activate.ps1
pytest
```
- Tests ignore archived/experimental folders by default.

## Getting unstuck
- Re-run `scripts\run_dev.ps1` to reinstall dependencies if needed.
- If Docker is used, ensure Docker Desktop is running.
- Share any error text you see; it will help us guide the next step. 