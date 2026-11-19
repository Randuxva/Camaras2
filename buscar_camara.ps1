# Busca en la red local los equipos con el puerto de video (554 o 8554) abierto: la camara.
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '169.*' -and $_.IPAddress -ne '127.0.0.1' } | Select-Object -First 1).IPAddress
$base = ($ip -split '\.')[0..2] -join '.'
Write-Host "Esta PC es $ip. Buscando en $base.1 - $base.254 ..."
$tareas = foreach ($i in 1..254) { foreach ($p in 554, 8554) {
    $c = New-Object Net.Sockets.TcpClient
    [pscustomobject]@{ Ip = "$base.$i"; Puerto = $p; Cliente = $c; Tarea = $c.ConnectAsync("$base.$i", $p) } } }
Start-Sleep -Seconds 4
$hallados = @()
foreach ($t in $tareas) { if ($t.Tarea.Status -eq 'RanToCompletion' -and $t.Cliente.Connected) { $hallados += "$($t.Ip):$($t.Puerto)" }; $t.Cliente.Close() }
if ($hallados) { Write-Host "Posibles camaras (IP:puerto):"; $hallados | ForEach-Object { Write-Host "  $_" } }
else { Write-Host "No encontre nada. Mira la IP en DMSS o en la pagina del router." }
