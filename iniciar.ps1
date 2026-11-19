# Arranca el agente de la cámara.
Set-Location $PSScriptRoot
if (-not (Test-Path .\venv\Scripts\python.exe)) { Write-Host "Primero corré .\instalar.ps1"; exit 1 }
.\venv\Scripts\python.exe main.py
