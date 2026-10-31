# ==============================================================================
# TradeSense v6 — Turnkey Laptop Live Stream Tunnel (Cloudflare Quick Tunnel)
# ==============================================================================
# Shares your local running TradeSense instance (RTX 4070 GPU + SQLite) publicly
# with a free, temporary HTTPS URL (e.g. https://xxx.trycloudflare.com)
#
# No login, account, port-forwarding, or administrator privileges required!
# ==============================================================================

$Port = 8501
$ToolsDir = Join-Path $PSScriptRoot "..\tools"
if (-not (Test-Path $ToolsDir)) {
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
}

$CloudflaredPath = Join-Path $ToolsDir "cloudflared.exe"

# 1. Download portable cloudflared if not present
if (-not (Test-Path $CloudflaredPath)) {
    Write-Host ">>> Downloading portable Cloudflare Quick Tunnel binary to tools/..." -ForegroundColor Cyan
    $Url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $Url -OutFile $CloudflaredPath -UseBasicParsing
    Write-Host ">>> Download complete." -ForegroundColor Green
}

# 2. Check if Streamlit is running on port 8501
$Listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $Listening) {
    Write-Host ">>> Warning: Streamlit is not detected on port $Port." -ForegroundColor Yellow
    Write-Host ">>> Starting Streamlit in background..." -ForegroundColor Yellow
    $env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..").Path
    $PythonExe = "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe"
    $AppPath = Join-Path (Resolve-Path "$PSScriptRoot\..").Path "src\ui\app.py"
    Start-Process $PythonExe -ArgumentList "-m streamlit run `"$AppPath`" --server.port $Port --server.headless true --theme.base dark"
    Start-Sleep -Seconds 3
}

# 3. Launch Cloudflare Tunnel
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " TradeSense Live Sharing Tunnel (Port $Port)" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Starting tunnel... Look for the public https://*.trycloudflare.com URL below:" -ForegroundColor White

& $CloudflaredPath tunnel --url "http://localhost:$Port"
