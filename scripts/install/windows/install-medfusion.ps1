param(
    [switch]$SkipDependencySync,
    [switch]$BuildFrontend,
    [switch]$VerifyStart,
    [int]$VerifyPort = 18080,
    [switch]$RunSmoke
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args
    )

    Write-Host "+ $Exe $($Args -join ' ')"
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Exe $($Args -join ' ')"
    }
}

function Require-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Hint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command '$Name'. $Hint"
    }
}

function Test-PythonVersionViaUv {
    $version = & uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query Python version via 'uv run python'."
    }

    $parts = $version.Trim().Split(".")
    if ($parts.Length -lt 2) {
        throw "Unexpected Python version string: $version"
    }

    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
        throw "Python 3.11+ is required. Current: $version"
    }
}

function Test-WebHealth {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 90
    )

    $url = "http://127.0.0.1:$Port/health"
    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 1
            continue
        }
        Start-Sleep -Seconds 1
    }

    throw "Timed out waiting for Web health endpoint: $url"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..\..\..")
Set-Location $repoRoot

Write-Step "Environment checks"
Require-Command -Name "uv" -Hint "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
Test-PythonVersionViaUv

if (-not $SkipDependencySync) {
    Write-Step "Install project dependencies (uv sync --extra dev --extra web)"
    Invoke-Checked -Exe "uv" -Args @("sync", "--extra", "dev", "--extra", "web")
}

if ($BuildFrontend) {
    Write-Step "Build frontend static assets"
    Require-Command -Name "npm" -Hint "Install Node.js and npm first."
    Push-Location (Join-Path $repoRoot "web\frontend")
    try {
        Invoke-Checked -Exe "npm" -Args @("install")
        Invoke-Checked -Exe "npm" -Args @("run", "build")
        $distPath = Join-Path $repoRoot "web\frontend\dist\*"
        $targetPath = Join-Path $repoRoot "med_core\web\static"
        Copy-Item -Path $distPath -Destination $targetPath -Recurse -Force
    }
    finally {
        Pop-Location
    }
}

Write-Step "Verify MedFusion CLI entrypoint"
Invoke-Checked -Exe "uv" -Args @("run", "medfusion", "--version")

if ($VerifyStart) {
    Write-Step "Verify medfusion start"
    $outputDir = Join-Path $repoRoot "outputs"
    if (-not (Test-Path -LiteralPath $outputDir)) {
        New-Item -Path $outputDir -ItemType Directory | Out-Null
    }
    $logPath = Join-Path $outputDir "install_verify_windows.log"

    $process = Start-Process `
        -FilePath "uv" `
        -ArgumentList @("run", "medfusion", "start", "--host", "127.0.0.1", "--port", "$VerifyPort", "--no-browser") `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $logPath `
        -PassThru

    try {
        Test-WebHealth -Port $VerifyPort -TimeoutSeconds 90
        Write-Host "Web health check passed on port $VerifyPort."
    }
    finally {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
        }
    }
}

if ($RunSmoke) {
    Write-Step "Run Windows local release smoke"
    Invoke-Checked -Exe "uv" -Args @("run", "python", "scripts/release_smoke.py", "--mode", "local")
}

Write-Step "Install complete"
Write-Host "Repository root: $repoRoot"
Write-Host "Next recommended command:"
Write-Host "  uv run medfusion start"
