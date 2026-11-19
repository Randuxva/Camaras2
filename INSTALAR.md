# Instalar el agente en una PC con Windows

## Antes de empezar
1. **Python 3.14** instalado (`python --version`). Si no está: `winget install Python.Python.3.14`,
   y después cerrar y volver a abrir la terminal.
2. **Git** para bajar este repositorio (`winget install Git.Git`), o bajarlo como ZIP desde GitHub.
3. **Internet** la primera vez: se bajan los modelos de patentes (~80 MB, a `%USERPROFILE%\.cache`).
4. Elegir la carpeta FINAL antes de instalar (no moverla después). Evitar OneDrive y rutas con tildes o espacios.
5. **La cámara tiene que ser visible desde esta PC**: estar en la misma red que la cámara (o tener acceso por VPN).
   Probar con VLC: `rtsp://usuario:clave@IP_DE_LA_CAMARA:554/cam/realmonitor?channel=1&subtype=0`

## 1. Bajar el código
```bash
cd /d "%USERPROFILE%\Documents" && git clone https://github.com/datafire985-a11y/camara.git && cd camara
```

## 2. Instalar (una vez, 10 a 20 minutos)
```bash
cd /d "%USERPROFILE%\Documents\camara" && powershell -ExecutionPolicy Bypass -File .\instalar.ps1
```

## 3. Crear el `.env` (configuración y claves)
El repositorio **no trae claves**. Copiar el ejemplo y completarlo:
```bash
cd /d "%USERPROFILE%\Documents\camara" && copy .env.example .env && notepad .env
```
Completar como mínimo:
- Cámara: `DAHUA_HOST` (IP), `DAHUA_RTSP_PORT` (554), `DAHUA_USER` y `DAHUA_PASS`.
  Si no se conoce la IP: `.\buscar_camara.ps1` la busca en la red local.
- Nube: `SUPABASE_URL`, `SUPABASE_KEY`, `WEBHOOK_URL`, `ANPR_AGENT_SECRET`, `DEFAULT_SUCURSAL_ID`. Los da el
  administrador del sistema. **No se suben a GitHub ni se pegan en chats.**
- `CAMERA_ID`: un nombre para esta cámara.
- `ROI_SOLO_CRUCES="true"`: regla estricta del portón (solo el cruce de línea genera aviso).

> **Importante para pruebas:** si esta PC usa las MISMAS claves de la nube que la PC del estacionamiento y las dos
> corren a la vez, **cada auto se registra dos veces**. Para probar sin tocar el sistema real, dejar `SUPABASE_URL`
> vacío: el agente funciona igual y guarda los eventos solo en esta PC (`offline_queue.db`).

## 4. Probar y arrancar
Pruebas (debe decir `OK`):
```bash
cd /d "%USERPROFILE%\Documents\camara" && venv\Scripts\python.exe -m unittest discover -s tests > pruebas.txt 2>&1 & findstr /B /C:"Ran " /C:"OK" /C:"FAILED" pruebas.txt
```
Arrancar con la ventana abierta (si se cierra, se detiene):
```bash
cd /d "%USERPROFILE%\Documents\camara" && powershell -ExecutionPolicy Bypass -File .\iniciar.ps1
```
Con `SHOW_DISPLAY="true"` en el `.env` se abre una ventana con las líneas y los vehículos (tecla `q` para salir).

En segundo plano (sin ventana; el registro queda en `logs\agente.log`):
```bash
cd /d "%USERPROFILE%\Documents\camara" && powershell -NoProfile -Command "Start-Process -FilePath '.\venv\Scripts\python.exe' -ArgumentList 'main.py' -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput salida.txt -RedirectStandardError errores.txt"
```
Detenerlo:
```bash
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
```

## Recomendaciones
- Que la PC no se suspenda: `powercfg /change standby-timeout-ac 0`.
- Mejor la cámara por cable de red que por Wi-Fi: con Wi-Fi se pierden partes del video.
