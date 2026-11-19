# Cambia en .env la direccion de la camara (IP publica -> IP local) sin tocar nada mas.
# Uso:  .\configurar_camara.ps1            (pregunta la IP y el puerto)
param([string]$Ip, [string]$Puerto, [string]$ArchivoEnv = (Join-Path $PSScriptRoot ".env"))
if (-not $Ip) { $Ip = Read-Host "IP local de la camara (ej. 192.168.1.50)" }
if ($Ip -notmatch '^(\d{1,3}\.){3}\d{1,3}$') { Write-Host "Esa IP no es valida."; exit 1 }
if (-not $Puerto) { $Puerto = Read-Host "Puerto RTSP (Enter = 554)" }
if (-not $Puerto) { $Puerto = "554" }
if (-not (Test-Path $ArchivoEnv)) { Write-Host "No encuentro $ArchivoEnv"; exit 1 }


$t = [IO.File]::ReadAllText($ArchivoEnv)
$t = [regex]::Replace($t, '(?m)^(CAMERA_SOURCE="[^"]*@)[^/@"]+(/[^"]*")', { param($m) $m.Groups[1].Value + "${Ip}:${Puerto}" + $m.Groups[2].Value })
$t = [regex]::Replace($t, '(?m)^DAHUA_HOST=".*"', "DAHUA_HOST=`"$Ip`"")
$t = [regex]::Replace($t, '(?m)^DAHUA_RTSP_PORT=".*"', "DAHUA_RTSP_PORT=`"$Puerto`"")
[IO.File]::WriteAllText($ArchivoEnv, $t, (New-Object Text.UTF8Encoding($false)))
Write-Host "Listo: .env ahora apunta a ${Ip}:${Puerto} (CAMERA_ID no se toco)."

$c = New-Object Net.Sockets.TcpClient
$ok = $c.ConnectAsync($Ip, [int]$Puerto).Wait(3000) -and $c.Connected
$c.Close()
if ($ok) { Write-Host "La camara responde en ese puerto. Siguiente: .\iniciar.ps1 (cuando toque el cambio)." }
else { Write-Host "La camara NO responde en ${Ip}:${Puerto}. Proba con otro puerto (554 o 8554) o revisa la IP con .\buscar_camara.ps1" }
