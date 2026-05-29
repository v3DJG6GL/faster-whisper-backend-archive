# Install (or reinstall) the faster-whisper-backend as a Windows Service
# via WinSW (https://github.com/winsw/winsw, v2.12.0). Run from any
# PowerShell prompt:
#   .\install-service.ps1
#   .\install-service.ps1 -WithConvert     # also install ~2 GB of HF->CT2 deps
# (Self-elevates to admin via UAC if not already running elevated.)
#
# WinSW.exe is auto-downloaded into this folder if missing. No package
# manager (choco/scoop/winget) required.
#
# Conversion extras (HF->CT2 auto-conversion): the script asks at install
# time whether to install them when they're missing -- install them later
# from /settings's AUTO_CONVERT_HF_MODELS toggle requirement. Already-
# installed extras are detected and the prompt is skipped silently.
# -WithConvert forces install without prompting (CI / scripted use).
#
# Pre-flight migration: if a service named "WhisperAPI" already exists
# (e.g. installed via the previous NSSM-based script), it is stopped and
# removed before installing the WinSW-managed replacement. The legacy
# nssm.exe is also removed.
#
# Run-as account: stays at the WinSW default (LocalSystem). Issue
# winsw#1136 reports SCM access denied on clean exit when running as a
# non-admin account; LocalSystem dodges it. Don't change this without
# reading the issue.

[CmdletBinding()]
param(
    [switch]$WithConvert
)

$ErrorActionPreference = "Stop"

# --- elevate to admin if needed ---------------------------------------------
# WinSW install/configuration/start all require administrator rights.
# Without elevation the SCM rejects the registration and the service ends
# up in a broken state.
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs admin rights. Triggering UAC..." -ForegroundColor Yellow
    $relaunchArgs = @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    # Forward switches across the UAC re-launch so $WithConvert isn't lost
    # when the script restarts under elevation.
    if ($WithConvert) { $relaunchArgs += "-WithConvert" }
    Start-Process powershell -Verb RunAs -ArgumentList $relaunchArgs
    exit
}

# --- locate paths ------------------------------------------------------------
$RepoDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python      = Join-Path $RepoDir "venv\Scripts\python.exe"
$MainPy      = Join-Path $RepoDir "main.py"
$LogsDir     = Join-Path $RepoDir "logs"
$ServiceName = "WhisperAPI"
$WinSWExe    = Join-Path $RepoDir "$ServiceName.exe"
$WinSWXml    = Join-Path $RepoDir "$ServiceName.xml"
$LegacyNssm  = Join-Path $RepoDir "nssm.exe"

if (-not (Test-Path $MainPy))  { throw "main.py not found: $MainPy" }
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

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

    # Purge the HTTP cache after the pip upgrade. Older pip versions store
    # cache entries in a format the upgraded pip can't deserialize, producing
    # a "Cache entry deserialization failed" warning per package -- harmless
    # (pip re-downloads) but very noisy. Ignore failures: a clean cache is
    # not load-bearing.
    & $Python -m pip cache purge 2>&1 | Out-Null

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

# --- optional: install HF->CT2 conversion extras ----------------------------
# Required only when AUTO_CONVERT_HF_MODELS=true in /settings. Adds ~2 GB
# (transformers + torch + accelerate).
#
# Decision tree:
#   -WithConvert flag -> install without prompting (CI / scripted use).
#   Already installed -> silent skip (idempotent re-run).
#   Otherwise        -> interactive y/N prompt.

function Test-ConvertDepsInstalled {
    # Probe by trying to import all three deps in the venv. Exit code 0 = all
    # present. Python writes the ImportError traceback to stderr; under
    # $ErrorActionPreference=Stop, PowerShell turns that into a terminating
    # NativeCommandError before the 2>$null redirect kicks in. Relax the
    # pref locally and merge stderr->stdout so the probe stays silent
    # regardless of import outcome.
    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python -c "import torch, transformers, accelerate" 2>&1 | Out-Null
    } finally {
        $ErrorActionPreference = $oldPref
    }
    return ($LASTEXITCODE -eq 0)
}

function Install-ConvertDeps {
    $convertReq = Join-Path $RepoDir "requirements-convert.txt"
    if (-not (Test-Path $convertReq)) {
        Write-Host "WARNING: requirements-convert.txt not found at $convertReq" -ForegroundColor Yellow
        return
    }
    Write-Host "Installing conversion extras (transformers + torch + accelerate, ~2 GB)..." -ForegroundColor Cyan
    & $Python -m pip install -r $convertReq
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-convert.txt failed (exit $LASTEXITCODE)" }
    Write-Host "Conversion extras installed. AUTO_CONVERT_HF_MODELS=true will now work." -ForegroundColor Green
}

if ($WithConvert) {
    Install-ConvertDeps
} elseif (Test-ConvertDepsInstalled) {
    Write-Host "Conversion extras already installed (transformers + torch + accelerate)." -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "Optional: HF->CT2 conversion extras are NOT installed." -ForegroundColor Yellow
    Write-Host "  These let the backend auto-convert HuggingFace transformers"
    Write-Host "  Whisper checkpoints (e.g. Flurin17/whisper-large-v3-turbo-swiss-german)"
    Write-Host "  to CTranslate2 format on first load. Footprint: ~2 GB (torch + transformers"
    Write-Host "  + accelerate). Required only when AUTO_CONVERT_HF_MODELS=true in /settings."
    $reply = Read-Host "Install conversion extras now? [y/N]"
    if ($reply -match '^(y|yes)$') {
        Install-ConvertDeps
    } else {
        Write-Host "Skipped. Re-run with -WithConvert later if you change your mind." -ForegroundColor DarkGray
    }
}

# --- pre-flight: stop + remove any existing WhisperAPI service -------------
# Handles BOTH the legacy NSSM-installed service AND a previous WinSW install
# (re-running this script). WinSW's `install` is not idempotent, so we always
# drop and re-register.
function Wait-ServiceGone($name, $timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Service -Name $name -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne "Stopped") {
        Write-Host "Stopping existing $ServiceName service..."
        try {
            Stop-Service -Name $ServiceName -Force -ErrorAction Stop
        } catch {
            # Stop-Service can throw if the service is in a transient state;
            # the polling loop below catches up regardless.
            Write-Host "  (stop signal sent; waiting for service to settle)" -ForegroundColor DarkGray
        }
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            $cur = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
            if (-not $cur -or $cur.Status -eq "Stopped") { break }
            Start-Sleep -Milliseconds 500
        }
    }

    Write-Host "Removing existing $ServiceName service..."
    # Prefer the right tool for whichever supervisor is currently in place.
    if (Test-Path $WinSWExe) {
        & $WinSWExe uninstall 2>&1 | Out-Null
    } elseif (Test-Path $LegacyNssm) {
        & $LegacyNssm remove $ServiceName confirm 2>&1 | Out-Null
    } else {
        # Bare SCM delete works regardless of which supervisor registered it.
        & sc.exe delete $ServiceName | Out-Null
    }

    if (-not (Wait-ServiceGone $ServiceName 15)) {
        Write-Host "WARNING: '$ServiceName' is still registered after removal." -ForegroundColor Yellow
        Write-Host "  Close any open services.msc / Event Viewer windows and retry,"
        Write-Host "  or reboot to clear the SCM 'marked for deletion' state."
        throw "service removal did not complete in 15 s"
    }
}

# Kill orphan python.exe processes rooted in this repo. WinSW's 30 s stop
# timeout often elapses during the ~minute-long model preload, then
# sc.exe delete removes the service entry without actually terminating
# python.exe. The orphan keeps port 8000 + the log file open, so the
# fresh install runs alongside it — old code keeps serving while the new
# process never wins the port. Without this cleanup the deploy is
# silently broken: code on disk says version N, behavior says N-1.
$orphans = Get-Process -ErrorAction SilentlyContinue |
    Where-Object {
        try { $_.Path -and $_.Path.StartsWith($RepoDir, [StringComparison]::OrdinalIgnoreCase) }
        catch { $false }
    }
if ($orphans) {
    Write-Host "Killing $($orphans.Count) orphan python.exe process(es) from $RepoDir..." -ForegroundColor Yellow
    $orphans | Stop-Process -Force -ErrorAction SilentlyContinue
    # Brief settle so the OS releases the port + log-file handle before
    # the new service starts. 2 s is overkill on a normal machine but
    # cheap insurance against a slow handle-close.
    Start-Sleep -Seconds 2
}

# One-time cleanup: drop the legacy NSSM binary if it's still in the repo dir.
if (Test-Path $LegacyNssm) {
    Write-Host "Removing legacy nssm.exe (no longer used)..."
    Remove-Item -Force $LegacyNssm
}

# --- pick the right WinSW binary -------------------------------------------
# WinSW v2.12.0 ships several executables:
#   WinSW.NET461.exe (~640 KB) -- requires .NET Framework 4.6.1+, OS-tracked & patched
#   WinSW-x64.exe    (~17 MB)  -- bundles .NET Core 3.1 (EOL Dec 2022, sec scanners flag it)
# .NET 4.8 ships preinstalled on Windows 10 1903+ / Win11 / Server 2022, so
# .NET461 is the right pick for our supported targets. Fall back to the
# bundled-runtime build only if 4.6.1 isn't available.
$WinSWVersion = "v2.12.0"
$net4Release  = (Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" `
                 -Name Release -ErrorAction SilentlyContinue).Release
if ($net4Release -ge 394254) {
    $WinSWAsset = "WinSW.NET461.exe"
    Write-Host "Using $WinSWAsset (host has .NET Framework 4.6.1+: Release=$net4Release)" -ForegroundColor DarkGray
} else {
    $WinSWAsset = "WinSW-x64.exe"
    Write-Host "Using $WinSWAsset (host lacks .NET 4.6.1; falling back to bundled .NET Core 3.1 build)" -ForegroundColor Yellow
}

# --- download WinSW.exe if missing -----------------------------------------
if (-not (Test-Path $WinSWExe)) {
    $url = "https://github.com/winsw/winsw/releases/download/$WinSWVersion/$WinSWAsset"
    Write-Host "Downloading WinSW $WinSWVersion ($WinSWAsset)..." -ForegroundColor Cyan
    Write-Host "  $url" -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $url -OutFile $WinSWExe -UseBasicParsing
    $sz = (Get-Item $WinSWExe).Length
    if ($sz -lt 100KB) {
        Remove-Item -Force $WinSWExe
        throw "WinSW download produced a $sz-byte file (expected >100 KB) - download failed"
    }
    Write-Host "WinSW.exe placed at: $WinSWExe ($([math]::Round($sz/1KB)) KB)" -ForegroundColor Green
}

# --- write WhisperAPI.xml --------------------------------------------------
# Always overwritten so edits to this here-string actually take effect on
# re-install. %BASE% is a WinSW built-in that resolves to the directory
# containing WhisperAPI.exe -- portable across deployments.
#
# Self-restart contract: <onfailure action="restart"/> is defense-in-depth
# for crashes. The "real" graceful-restart path (admin WebUI button) is
# driven by restart_service.py spawning `WhisperAPI.exe restart!` BEFORE
# os._exit(0) -- v2's <onfailure> semantics on exit-0 are unreliable.
$xml = @"
<?xml version="1.0" encoding="UTF-8"?>
<service>
  <id>$ServiceName</id>
  <name>$ServiceName</name>
  <description>Self-hosted faster-whisper API (CH-DE dictation)</description>

  <executable>%BASE%\venv\Scripts\python.exe</executable>
  <arguments>%BASE%\main.py</arguments>
  <workingdirectory>%BASE%</workingdirectory>
  <startmode>Automatic</startmode>

  <!-- Stop: send Ctrl-C, wait up to 30 s, then TerminateProcess.
       Uvicorn handles SIGINT cleanly; we run SERVER_WORKERS=1 so signaling
       the parent first is correct. WinSW v2 does NOT send WM_CLOSE/WM_QUIT
       to console apps, so the NSSM AppStopMethodSkip workaround isn't needed. -->
  <stoptimeout>30 sec</stoptimeout>
  <stopparentprocessfirst>true</stopparentprocessfirst>

  <!-- Crash-restart with back-off. Graceful restart (admin WebUI) goes
       through `WhisperAPI.exe restart!` from restart_service.py instead. -->
  <onfailure action="restart" delay="2 sec"/>
  <resetfailure>1 hour</resetfailure>

  <!-- WinSW writes WhisperAPI.out.log + WhisperAPI.err.log SEPARATELY
       (basenames not configurable in v2). sizeThreshold is in KB, NOT bytes. -->
  <logpath>%BASE%\logs</logpath>
  <log mode="roll-by-size">
    <sizeThreshold>10240</sizeThreshold>
    <keepFiles>8</keepFiles>
  </log>

  <env name="WHISPER_LOG_FILE" value="%BASE%\logs\whisper.log"/>
  <!-- To enable the admin WebUI via env (alternative: set ADMIN_UI_ENABLED in config.py),
       uncomment and edit the lines below, then re-run this install script.
       Generate a strong token with:
         [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
  <env name="WHISPER_ADMIN_UI" value="1"/>
  <env name="WHISPER_ADMIN_TOKEN" value="paste-32-byte-base64-here"/>
  -->
</service>
"@
Set-Content -Path $WinSWXml -Value $xml -Encoding UTF8
Write-Host "WhisperAPI.xml written: $WinSWXml" -ForegroundColor DarkGray

# --- install + start --------------------------------------------------------
# WinSW writes UTF-16 console output; PowerShell's default decode produces
# garbled "W h i s p e r A P I" text. Scope a UTF-16 OutputEncoding only
# around the WinSW calls. WinSW also writes informational status to stderr
# ("Service is starting..."), which PowerShell turns into NativeCommandError
# under $ErrorActionPreference=Stop -- relax that too.
function Invoke-WinSW {
    $oldEncoding = [Console]::OutputEncoding
    $oldPref     = $ErrorActionPreference
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $ErrorActionPreference    = "Continue"
    try {
        & $WinSWExe @args
    } finally {
        [Console]::OutputEncoding = $oldEncoding
        $ErrorActionPreference    = $oldPref
    }
}

Write-Host "Installing $ServiceName service via WinSW..."
Invoke-WinSW install
Write-Host "Starting $ServiceName service..."
Invoke-WinSW start

# --- verify -----------------------------------------------------------------
Start-Sleep -Seconds 3
$final = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $final) {
    Write-Host ""
    Write-Host "FAILED: service is not registered after install." -ForegroundColor Red
    Write-Host "Re-run this script in an elevated PowerShell prompt."
    exit 1
}
# Poll up to 30 s for Running -- model preload at startup can push past 3 s.
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $final.Status -ne "Running") {
    Start-Sleep -Milliseconds 500
    $final = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $final) { break }
}
if (-not $final -or $final.Status -ne "Running") {
    $statusStr = if ($final) { $final.Status } else { "missing" }
    Write-Host ""
    Write-Host "WARNING: service status is '$statusStr', expected 'Running'." -ForegroundColor Yellow
    Write-Host "Check service logs:"
    Write-Host "  $LogsDir\$ServiceName.err.log"
    Write-Host "  $LogsDir\$ServiceName.out.log"
    Write-Host "  $LogsDir\whisper.log"
    exit 1
}

Write-Host ""
Write-Host "Done. Service is running." -ForegroundColor Green
Write-Host "  API:        http://localhost:8000/v1/audio/transcriptions"
Write-Host "  Live logs:  http://localhost:8000/logs"
Write-Host "  Stats:      http://localhost:8000/stats"
Write-Host "  Admin UI:   http://localhost:8000/settings  (only when ADMIN_UI_ENABLED=True in config.py, or WHISPER_ADMIN_UI=1)"
Write-Host "  App log:    $LogsDir\whisper.log"
Write-Host "  Stdout/err: $LogsDir\$ServiceName.out.log  /  $LogsDir\$ServiceName.err.log"
Write-Host ""
Write-Host "Useful commands (work from any directory):"
Write-Host "  Restart-Service WhisperAPI"
Write-Host "  Stop-Service WhisperAPI"
Write-Host "  Start-Service WhisperAPI"
Write-Host "  Get-Service WhisperAPI"
Write-Host "  Get-Content -Wait '$LogsDir\whisper.log'"
