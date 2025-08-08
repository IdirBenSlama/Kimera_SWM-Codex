Param(
    [switch]$WithInfra,
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }

try {
    # Move to repo root (this script is under scripts/)
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location (Join-Path $ScriptDir "..")

    Write-Info "Project root: $(Get-Location)"

    # Ensure Python
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) { $pythonCmd = Get-Command py -ErrorAction SilentlyContinue }
    if (-not $pythonCmd) { throw "Python is not in PATH. Install Python 3.11+ and reopen PowerShell." }

    # Create venv if missing
    if (-not (Test-Path ".venv")) {
        Write-Info "Creating virtual environment (.venv)"
        & $pythonCmd.Source -m venv .venv
    }

    # Path to venv python
    $venvPython = Join-Path ".venv" "Scripts/python.exe"
    if (-not (Test-Path $venvPython)) { throw "Virtualenv python not found at $venvPython" }

    # Upgrade pip
    Write-Info "Upgrading pip"
    & $venvPython -m pip install --upgrade pip | Out-Null

    # Ensure uvicorn is available
    Write-Info "Ensuring uvicorn is installed"
    & $venvPython -m pip install -q uvicorn | Out-Null

    # Install consolidated requirements (tolerant if some files missing)
    $reqRoot = "requirements_consolidated"
    $reqs = @("base.txt", "api.txt") | ForEach-Object { Join-Path $reqRoot $_ }
    foreach ($req in $reqs) {
        if (Test-Path $req) {
            Write-Info "Installing requirements: $req"
            & $venvPython -m pip install -r $req | Out-Null
        } else {
            Write-Warn "Requirements file not found: $req (skipping)"
        }
    }

    # Environment variables
    if (-not $env:KIMERA_ENV) { $env:KIMERA_ENV = "development" }
    # Enforce clean import boundary: only allow imports from src/
    $env:PYTHONPATH = (Join-Path (Get-Location) 'src')
    Write-Info "KIMERA_ENV=$($env:KIMERA_ENV)"
    Write-Info "PYTHONPATH=$($env:PYTHONPATH)"

    if ($WithInfra) {
        # Start essential infra via Docker if available
        $docker = Get-Command docker -ErrorAction SilentlyContinue
        if ($docker) {
            $composeFile = "config/docker/docker-compose.yml"
            if (Test-Path $composeFile) {
                Write-Info "Starting Docker services (Postgres, Redis, Prometheus)"
                docker compose -f $composeFile up -d postgres redis prometheus | Out-Null
                if (-not $env:DATABASE_URL) {
                    $env:DATABASE_URL = "postgresql://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm"
                }
            } else {
                Write-Warn "Compose file not found at $composeFile; skipping infra startup"
            }
        } else {
            Write-Warn "Docker not found; skipping infra startup"
        }
    } else {
        if (-not $env:DATABASE_URL) {
            # Default to SQLite if DATABASE_URL not set
            $env:DATABASE_URL = "sqlite:///kimera_swm.db"
        }
    }

    Write-Info "DATABASE_URL=$($env:DATABASE_URL)"

    # Run FastAPI (single entrypoint)
    $uvicornTarget = "src.api.app_main:app"
    Write-Info ("Starting API: http://{0}:{1} (Press Ctrl+C to stop)" -f $BindHost, $Port)
    & $venvPython -m uvicorn $uvicornTarget --host $BindHost --port $Port --reload

    Write-Ok "API stopped"
}
catch {
    Write-Err $_
    exit 1
}
