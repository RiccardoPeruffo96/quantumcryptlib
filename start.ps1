# start.ps1

Write-Output "Framework Kanitchar-Huber started with $(python --version)"

# params: 
# "--d", type=int, default=4, help="Qudit dimension (d >= 2)"
# "--qberZ", type=float, default=0.05, help="QBER (Quantum Bit Error Rate) on Z (0 <= qberZ <= 1)"
# "--tc", type=int, default=100000, help="Total coincidences (number of photons exchanged in a small slice of time) (tc >= 1)"
# "--vX", type=float, default=0.88, help="Total visibility (how much noise or interception are in the channel, best near 1.0) (0 <= visibilityX <= 1)"

Write-Output "Framework Kanitchar-Huber started with $(python --version)" > out\info.log
python .\src\main.py --d 4 >> out\info.log
#python .\src\main.py --d 4 --qberZ 0.001 --vX 0.999 >> out\info.log