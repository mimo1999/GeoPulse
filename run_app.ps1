# GeoPulse one-click launcher: starts backend (FastAPI/uvicorn) + frontend (Streamlit)
# and opens the dashboard in your browser.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "== GeoPulse launcher ==" -ForegroundColor Cyan

# --- Sanity check: Postgres reachable on the configured port ---
$pgOk = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet -WarningAction SilentlyContinue
if (-not $pgOk) {
    Write-Host "WARNING: Nothing is listening on localhost:5432 (Postgres)." -ForegroundColor Yellow
    Write-Host "The backend will start, but data-backed endpoints will fail until Postgres is running." -ForegroundColor Yellow
}

# --- Start backend (FastAPI/uvicorn) in its own window ---
Write-Host "Starting backend on http://localhost:8000 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
) -WindowStyle Normal

# --- Start frontend (Streamlit) in its own window ---
Write-Host "Starting frontend on http://localhost:8501 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; `$env:BACKEND_URL='http://localhost:8000'; streamlit run streamlit_app/app.py --server.port 8501"
) -WindowStyle Normal

# --- Wait for backend health check, then open the browser ---
Write-Host "Waiting for backend to become healthy..."
$healthy = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $healthy = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}

if ($healthy) {
    Write-Host "Backend is healthy." -ForegroundColor Green
} else {
    Write-Host "Backend did not report healthy within 30s - check its window for errors." -ForegroundColor Yellow
}

Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "GeoPulse is starting up:" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:8000  (docs at /docs)"
Write-Host "  Frontend: http://localhost:8501"
Write-Host ""
Write-Host "Two PowerShell windows were opened for backend/frontend logs. Close them to stop the app."
