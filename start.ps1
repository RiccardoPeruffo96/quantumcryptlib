# start.ps1

Write-Output "Framework Kanitchar-Huber started with $(python --version)"

Write-Output "Framework Kanitchar-Huber started with $(python --version)" > out\info.log
python .\src\main.py --d 4 >> out\info.log