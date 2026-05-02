# リポジトリ直下（キカガク）から PDF まで一括。latex に既にいる場合は二重 cd しない。
# 使い方: プロジェクトルートで  .\scripts\compile_final_report.ps1
#    または: 任意の場所から  & "D:\...\キカガク\scripts\compile_final_report.ps1"

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[1/3] assumed_supplementary_analysis (repo root: $RepoRoot)" -ForegroundColor Cyan
py -3 -m analytics.assumed_supplementary_analysis
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] lualatex (1st pass)" -ForegroundColor Cyan
Push-Location (Join-Path $RepoRoot "latex")
try {
    lualatex -interaction=nonstopmode final_report.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "[3/3] lualatex (2nd pass)" -ForegroundColor Cyan
    lualatex -interaction=nonstopmode final_report.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

Write-Host "Done: latex\final_report.pdf" -ForegroundColor Green
