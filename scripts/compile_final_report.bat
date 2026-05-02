@echo off
setlocal
cd /d "%~dp0.."

echo [1/3] assumed_supplementary_analysis (repo root: %CD%)
py -3 -m analytics.assumed_supplementary_analysis
if errorlevel 1 exit /b 1

echo [2/3] lualatex (1st pass)
pushd latex
lualatex -interaction=nonstopmode final_report.tex
if errorlevel 1 popd & exit /b 1

echo [3/3] lualatex (2nd pass)
lualatex -interaction=nonstopmode final_report.tex
set ERR=%ERRORLEVEL%
popd
if %ERR% neq 0 exit /b %ERR%

echo Done: latex\final_report.pdf
