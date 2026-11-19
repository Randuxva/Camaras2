# Crea el entorno de Python e instala las dependencias (versiones fijas).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-fijas.txt
Write-Host "Listo. Ahora: .\iniciar.ps1"
