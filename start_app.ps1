# start_app.ps1
# ==============================================================================
# SnakeGuard AI - Startup Script
# ==============================================================================
# Primary method to start the server for stable, normal usage.
# This script clears the port and starts the server in production-like mode.

$port = 8000
echo "----------------------------------------------------"
echo "SnakeGuard AI: Checking for processes on port $port..."

# Find the PID of the process using the target port
$proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

if ($proc) {
    echo "Found process $proc using port $port. Terminating for a clean start..."
    Stop-Process -Id $proc -Force
    Start-Sleep -Seconds 2
} else {
    echo "Port $port is clear."
}

echo "Starting SnakeGuard AI Server (Stable Mode)..."
echo "URL: http://localhost:8000"
echo "----------------------------------------------------"

# Set environment variables for consistent encoding and output
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"

# Run Uvicorn without --reload for maximum stability and lower resource usage
.\sb_env\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port $port
