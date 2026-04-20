param(
    [switch]$PurgeData,
    [switch]$RemoveNodeModules,
    [switch]$RemoveFrontendDist
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Yellow
}

function Remove-PathIfExists {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $PathValue)) {
        Write-Host "${Label}: skip (not found)"
        return
    }

    Write-Host "${Label}: remove $PathValue"
    Remove-Item -LiteralPath $PathValue -Recurse -Force
}

function Get-DefaultUserDataDir {
    if ($env:MEDFUSION_DATA_DIR -and $env:MEDFUSION_DATA_DIR.Trim().Length -gt 0) {
        return $env:MEDFUSION_DATA_DIR.Trim()
    }
    return (Join-Path $env:USERPROFILE ".medfusion")
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..\..\..")
Set-Location $repoRoot

Write-Step "Remove workspace runtime artifacts"
Remove-PathIfExists -PathValue (Join-Path $repoRoot ".venv") -Label ".venv"
Remove-PathIfExists -PathValue (Join-Path $repoRoot ".pytest_cache") -Label ".pytest_cache"
Remove-PathIfExists -PathValue (Join-Path $repoRoot ".mypy_cache") -Label ".mypy_cache"
Remove-PathIfExists -PathValue (Join-Path $repoRoot "outputs\install_verify_windows.log") -Label "install verify log"

if ($RemoveNodeModules) {
    Write-Step "Remove frontend node_modules"
    Remove-PathIfExists -PathValue (Join-Path $repoRoot "web\frontend\node_modules") -Label "web/frontend/node_modules"
}

if ($RemoveFrontendDist) {
    Write-Step "Remove frontend dist"
    Remove-PathIfExists -PathValue (Join-Path $repoRoot "web\frontend\dist") -Label "web/frontend/dist"
}

if ($PurgeData) {
    Write-Step "Purge MedFusion data directories"
    $userDataDir = Get-DefaultUserDataDir
    Remove-PathIfExists -PathValue $userDataDir -Label "user data dir"
    Remove-PathIfExists -PathValue (Join-Path $repoRoot "outputs") -Label "repo outputs"
    Remove-PathIfExists -PathValue (Join-Path $repoRoot "logs") -Label "repo logs"
    Remove-PathIfExists -PathValue (Join-Path $repoRoot "checkpoints") -Label "repo checkpoints"
}
else {
    Write-Step "Keep data mode"
    Write-Host "Data directories are preserved."
    Write-Host "Use -PurgeData for full data cleanup."
}

Write-Step "Uninstall complete"
Write-Host "Repository root cleaned runtime artifacts: $repoRoot"
Write-Host "Purge data mode: $($PurgeData.IsPresent)"
