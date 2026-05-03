# Install (or reinstall) the faster-whisper-backend as a Windows Service
# via NSSM. Run from any PowerShell prompt:
#   .\install-service.ps1
# (Self-elevates to admin via UAC if not already running elevated.)
#
# nssm.exe is auto-downloaded into this folder if it's not already on PATH.
# No package manager (choco/scoop/winget) required.
#
# Note: the service is registered under the name "WhisperAPI" for backward
# compatibility with existing installs. The repo folder may be renamed without
# touching the service name.

$ErrorActionPreference = "Stop"

# --- elevate to admin if needed ---------------------------------------------
# NSSM service install/configuration/start all require administrator rights.
# Without elevation the user sees a wall of "OpenService: Access is denied"
# errors and the service ends up STOPPED instead of RUNNING.
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs admin rights. Triggering UAC..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    exit
}

# --- locate paths ------------------------------------------------------------
$RepoDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python    = Join-Path $RepoDir "venv\Scripts\python.exe"
$MainPy    = Join-Path $RepoDir "main.py"
$LogsDir   = Join-Path $RepoDir "logs"
$StdoutLog = Join-Path $LogsDir "service-stdout.log"
$StderrLog = Join-Path $LogsDir "service-stderr.log"

if (-not (Test-Path $MainPy))  { throw "main.py not found: $MainPy" }

# --- bootstrap venv if missing ---------------------------------------------
# First-time install on a fresh clone has no venv yet. Create it inline using
# whatever Python the user has on PATH so the script is "clone -> run" with
# no manual prep. Idempotent: skipped entirely if venv already exists.
if (-not (Test-Path $Python)) {
    Write-Host "Python venv not found - bootstrapping..." -ForegroundColor Cyan

    # Prefer the PEP 397 launcher (py.exe), fall back to python / python3.
    $sysPy = $null
    foreach ($cand in @("py", "python", "python3")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) { $sysPy = $cmd.Source; break }
    }
    if (-not $sysPy) {
        throw "No Python found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ (check 'Add to PATH'), then re-run this script."
    }

    Write-Host "Creating venv with: $sysPy" -ForegroundColor Cyan
    & $sysPy -m venv (Join-Path $RepoDir "venv")
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
    if (-not (Test-Path $Python)) { throw "venv created but $Python still missing - check the Python install" }

    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }

    $reqFile = Join-Path $RepoDir "requirements.txt"
    if (Test-Path $reqFile) {
        Write-Host "Installing requirements (faster-whisper + CUDA wheels can take a few minutes)..." -ForegroundColor Cyan
        & $Python -m pip install -r $reqFile
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    } else {
        Write-Host "WARNING: requirements.txt not found at $reqFile" -ForegroundColor Yellow
    }
    Write-Host "venv ready." -ForegroundColor Green
}

# --- locate or download nssm.exe -------------------------------------------
function Get-NssmPath {
    # 1. Already on PATH?
    $cmd = Get-Command nssm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    # 2. Already downloaded into the repo dir?
    $localNssm = Join-Path $RepoDir "nssm.exe"
    if (Test-Path $localNssm) { return $localNssm }

    # 3. Download nssm-2.24 (~300 KB).
    Write-Host "nssm.exe not found - downloading from nssm.cc..." -ForegroundColor Cyan
    $url = "https://nssm.cc/release/nssm-2.24.zip"
    $zip = Join-Path $env:TEMP "nssm-2.24.zip"
    $extractDir = Join-Path $env:TEMP "nssm-2.24-extracted"

    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
    Expand-Archive -Path $zip -DestinationPath $extractDir -Force

    # Pick the right architecture: 64-bit OS gets win64\nssm.exe.
    $arch = if ([Environment]::Is64BitOperatingSystem) { "win64" } else { "win32" }
    $sourceExe = Get-ChildItem -Path $extractDir -Recurse -Filter "nssm.exe" |
                 Where-Object { $_.FullName -match "\\$arch\\" } |
                 Select-Object -First 1
    if (-not $sourceExe) { throw "Could not find $arch\nssm.exe inside the downloaded zip." }

    Copy-Item $sourceExe.FullName $localNssm
    Remove-Item $zip -Force
    Remove-Item -Recurse -Force $extractDir
    Write-Host "nssm.exe placed at: $localNssm" -ForegroundColor Green
    return $localNssm
}
$nssm = Get-NssmPath

# Helper to call nssm. Two PowerShell-specific gotchas with NSSM:
#   1. NSSM writes UTF-16 to the console; PowerShell defaults to ANSI when
#      decoding native command output, producing garbled text where every
#      other byte renders as a space ("W h i s p e r A P I").
#   2. NSSM writes informational status to stderr (e.g. "Unexpected status
#      STOP_PENDING"), which PowerShell turns into a NativeCommandError when
#      $ErrorActionPreference = "Stop", aborting the script.
# We scope both fixes to NSSM calls so the rest of the script keeps strict
# error handling.
function Invoke-Nssm {
    $oldEncoding = [Console]::OutputEncoding
    $oldPref     = $ErrorActionPreference
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $ErrorActionPreference    = "Continue"
    try {
        & $nssm @args
    } finally {
        [Console]::OutputEncoding = $oldEncoding
        $ErrorActionPreference    = $oldPref
    }
}

# --- ensure dirs ------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# --- remove existing service if present -------------------------------------
$svc = Get-Service -Name "WhisperAPI" -ErrorAction SilentlyContinue
if ($svc) {
    Write-Host "Stopping existing WhisperAPI service..."
    Invoke-Nssm stop WhisperAPI 2>&1 | Out-Null
    # Wait for the service to actually leave the running state. The `nssm stop`
    # call can return immediately while the SCM is still in StopPending.
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        $cur = Get-Service -Name "WhisperAPI" -ErrorAction SilentlyContinue
        if (-not $cur -or $cur.Status -eq "Stopped") { break }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "Removing existing WhisperAPI service..."
    Invoke-Nssm remove WhisperAPI confirm 2>&1 | Out-Null
    # Wait for the SCM to drop the registration before we re-install.
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Service -Name "WhisperAPI" -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }
}

# --- install ----------------------------------------------------------------
Write-Host "Installing WhisperAPI service..."
Invoke-Nssm install WhisperAPI $Python $MainPy
Invoke-Nssm set WhisperAPI AppDirectory $RepoDir
Invoke-Nssm set WhisperAPI Description "Self-hosted faster-whisper API (CH-DE dictation) for vowen.ai"
Invoke-Nssm set WhisperAPI Start SERVICE_AUTO_START
Invoke-Nssm set WhisperAPI AppStdout $StdoutLog
Invoke-Nssm set WhisperAPI AppStderr $StderrLog

# Self-restart contract for the admin WebUI's restart button: the python
# process exits cleanly on demand; NSSM relaunches it. AppExit=Restart is
# the NSSM default, but pin it so a future "nssm reset" doesn't change
# behavior. AppThrottle=1500 ms means NSSM only throttles re-launches that
# happen within 1.5 s of startup (defends against config-broke-the-boot
# loops). AppRestartDelay=1500 ms gives in-flight requests a chance to
# drain before the process dies.
Invoke-Nssm set WhisperAPI AppExit Default Restart
Invoke-Nssm set WhisperAPI AppThrottle 1500
# 10 MB rotation, keep recent files
Invoke-Nssm set WhisperAPI AppRotateFiles 1
Invoke-Nssm set WhisperAPI AppRotateOnline 1
Invoke-Nssm set WhisperAPI AppRotateBytes 10485760
# Don't drown CPU on graceful-stop; give 30s before kill.
Invoke-Nssm set WhisperAPI AppStopMethodConsole 30000
Invoke-Nssm set WhisperAPI AppStopMethodWindow 30000
Invoke-Nssm set WhisperAPI AppStopMethodThreads 30000

# Skip WM_CLOSE/WM_QUIT (bits 2|4 = 6); keep Ctrl-C (1) and TerminateProcess (8).
# uvicorn handles Ctrl-C, so the unused windowed signals just slow shutdown.
Invoke-Nssm set WhisperAPI AppStopMethodSkip 14
# Pause between auto-restarts. 1500 ms matches the admin WebUI's self-exit
# timer so in-flight requests can drain before the new process binds the
# port; also defends against fork-bombs on broken config.
Invoke-Nssm set WhisperAPI AppRestartDelay 1500

# --- environment for main.py ----------------------------------------------
# WHISPER_LOG_FILE: path the rotating logger writes to.
# Admin UI knobs (ADMIN_UI_ENABLED / ADMIN_TOKEN) live in config.py and can
# also be overridden here via WHISPER_ADMIN_UI=1 and WHISPER_ADMIN_TOKEN=...
# (env wins over config.py). Generate a strong token with:
#   [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
# Multiple env vars go on a single AppEnvironmentExtra line, space-separated.
Invoke-Nssm set WhisperAPI AppEnvironmentExtra "WHISPER_LOG_FILE=$LogsDir\whisper.log"
# Example with admin UI enabled via env (alternative: edit config.py directly):
#   Invoke-Nssm set WhisperAPI AppEnvironmentExtra `
#     "WHISPER_LOG_FILE=$LogsDir\whisper.log" `
#     "WHISPER_ADMIN_UI=1" `
#     "WHISPER_ADMIN_TOKEN=<paste-32-byte-base64-here>"

# --- start ------------------------------------------------------------------
Write-Host "Starting WhisperAPI service..."
Invoke-Nssm start WhisperAPI

Start-Sleep -Seconds 3
Invoke-Nssm status WhisperAPI | Out-Null

# Verify the service is actually running. NSSM's `status` only prints, it
# never sets an exit code, so we go to the SCM directly.
$final = Get-Service -Name "WhisperAPI" -ErrorAction SilentlyContinue
if (-not $final) {
    Write-Host ""
    Write-Host "FAILED: service is not registered after install." -ForegroundColor Red
    Write-Host "Re-run this script in an elevated PowerShell prompt."
    exit 1
}
if ($final.Status -ne "Running") {
    Write-Host ""
    Write-Host "WARNING: service status is '$($final.Status)', expected 'Running'." -ForegroundColor Yellow
    Write-Host "Check service logs:"
    Write-Host "  $StderrLog"
    Write-Host "  $StdoutLog"
    Write-Host "  $LogsDir\whisper.log"
    exit 1
}

Write-Host ""
Write-Host "Done. Service is running." -ForegroundColor Green
Write-Host "  API:        http://localhost:8000/v1/audio/transcriptions"
Write-Host "  Live logs:  http://localhost:8000/logs"
Write-Host "  Admin UI:   http://localhost:8000/config  (only when ADMIN_UI_ENABLED=True in config.py, or WHISPER_ADMIN_UI=1)"
Write-Host "  Log file:   $LogsDir\whisper.log"
Write-Host ""
Write-Host "Useful commands (work from any directory):"
Write-Host "  Restart-Service WhisperAPI"
Write-Host "  Stop-Service WhisperAPI"
Write-Host "  Start-Service WhisperAPI"
Write-Host "  Get-Service WhisperAPI"
Write-Host "  Get-Content -Wait '$LogsDir\whisper.log'"
