# start.ps1

Write-Output "Framework Kanitchar-Huber started with $(python --version)"

Write-Output "Framework Kanitchar-Huber started with $(python --version)" > result.txt
python .\src\main.py --d 4 >> result.txt