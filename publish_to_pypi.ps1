# DmDSLab PyPI Publishing Script
# Ultra-simple version without emoji or special characters

param([switch]$TestRepo)

Write-Host "DmDSLab PyPI Publishing Script" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green

# Configuration
$PackageName = "DmDSLab"
if ($TestRepo) {
    $PyPiRepo = "testpypi"
    $Token = $env:TEST_PYPI_TOKEN
} else {
    $PyPiRepo = "pypi"
    $Token = $env:PYPI_TOKEN
}

# Check token
if (-not $Token) {
    $TokenVar = if ($TestRepo) { "TEST_PYPI_TOKEN" } else { "PYPI_TOKEN" }
    Write-Host "ERROR: $TokenVar not set!" -ForegroundColor Red
    Write-Host "Set it with: Set-Variable -Name env:$TokenVar -Value 'your-token'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Publishing to: $PyPiRepo" -ForegroundColor Cyan

# Clean previous builds
Write-Host ""
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue

# Update tools
Write-Host "Updating build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip build twine
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Could not update tools!" -ForegroundColor Red
    exit 1
}

# Build package
Write-Host "Building package..." -ForegroundColor Yellow
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

# Check package
Write-Host "Checking package..." -ForegroundColor Yellow
twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Package check failed!" -ForegroundColor Red
    exit 1
}

# Publish
Write-Host "Publishing to $PyPiRepo..." -ForegroundColor Yellow
if ($TestRepo) {
    twine upload --repository testpypi --username __token__ --password $Token dist/*
} else {
    twine upload --username __token__ --password $Token dist/*
}

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! Package published successfully!" -ForegroundColor Green
    Write-Host "Check at: https://$PyPiRepo.org/project/$PackageName/" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "FAILED! Publication was not successful!" -ForegroundColor Red
    exit 1
}

# Cleanup
Write-Host "Cleaning up..." -ForegroundColor Yellow
Remove-Item -Recurse -Force build, *.egg-info -ErrorAction SilentlyContinue

Write-Host "Script completed!" -ForegroundColor Green