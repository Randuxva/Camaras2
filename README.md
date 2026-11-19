# Cámara del portón: agente de lectura de patentes

Programa en Python que mira el portón con una cámara IP (probado con Dahua DH-P5B-PV), detecta los vehículos,
decide si **entran o salen**, lee la patente y envía la foto y la lectura a la nube (Supabase), donde el sistema
registra la estadía.

```
Cámara (RTSP) ──► detección de vehículos (YOLO) + seguimiento ──► regla del portón (líneas A y B)
             └──► stream principal: recorte del vehículo ──► lector de patentes local ──► nube (cola offline)
```

## Instalación en una PC nueva (Windows)

Ver **[INSTALAR.md](INSTALAR.md)**: Python, dependencias, `.env` y primer arranque, paso a paso.

## Cómo decide ENTRADA / SALIDA

La cámara está dentro del garaje mirando a la calle. Se usa el borde inferior de cada vehículo (dónde apoya en el piso):

| Zona | Posición (`y` de 0 arriba a 1 abajo) | Qué es |
|---|---|---|
| CALLE | arriba de la línea B (`ROI_LINE_B_Y_RATIO`, 0,56) | tráfico y autos estacionados enfrente |
| PORTAL | entre B y A | portón y vereda |
| INTERIOR | abajo de la línea A (`ROI_LINE_A_Y_RATIO`, 0,68) | rampa, junto a la cámara |

**Regla estricta (`ROI_SOLO_CRUCES="true"`)**: el único aviso es el cruce.
- **ENTRADA**: el vehículo estuvo arriba de la línea B y llegó a la línea A.
- **SALIDA**: estuvo abajo de la línea A y llegó a la línea B.
- Un vehículo de arriba que solo toca la línea B (estacionado detrás de la reja, esperando en la puerta) **no genera
  nada**. Lo que se pierde de vista sin cruzar no se avisa: queda archivado con su foto en `quietos.db` y la carpeta
  `quietos/` (nunca se borra).
- Si el seguimiento de un vehículo se corta a mitad del portón (cambio de número del seguimiento por la reja o los
  faros), el tramo siguiente se une al anterior durante `ROI_UNION_SEC` y el cruce se evalúa sobre el recorrido completo.
- `"sombra"`: marca lo que archivaría pero envía todo (para probar). `"false"`: comportamiento anterior.

Si se mueve la cámara hay que recalibrar las líneas: `tools/dibujar_roi.py` dibuja las zonas sobre una captura.

## Lectura de patentes

En el equipo (`core/plate_reader.py`) se detecta la patente (`open-image-models`, YOLOv9 ONNX) y se lee
(`fast-plate-ocr`) en varios cuadros del mismo vehículo; las lecturas con formato argentino válido se combinan con un
voto por carácter. La foto del vehículo, las mejores vistas de la patente y la lectura local se envían a la nube.

## Rendimiento: dos streams

- **Sub-stream (640x360)**: detección y seguimiento en todos los cuadros que dé el equipo.
- **Stream principal**: recorte del vehículo en alta resolución con la posición del sub-stream en ese instante.
  `MAIN_DECODE_MODE="auto"` decodifica todos los cuadros si el equipo llega, y si no, solo los cuadros clave.

## Herramientas (`tools/`)

| Herramienta | Para qué |
|---|---|
| `record_streams.py` | Graba los dos streams de la cámara para pruebas |
| `replay.py` | Reproduce una grabación por el agente completo, sin red ni nube |
| `extraer_trayectorias.py` + `evaluar_reglas.py` | Compara reglas del portón sobre las mismas trayectorias |
| `dibujar_roi.py` | Dibuja las líneas y zonas sobre una captura de la cámara |
| `medir_rendimiento.py` | Mide en el equipo velocidad y memoria del detector y de los lectores (pide un video) |
| `benchmark_plates.py` | Compara lectores de patentes sobre una carpeta de fotos |
| `medir_desfase.py` | Mide el desfase entre el sub-stream y el principal |
| `requeue_failed.py` | Vuelve a encolar eventos que fallaron al subir |

## Pruebas

```bash
.\venv\Scripts\python.exe -m unittest discover -s tests
```

## Funcionamiento

- Sin internet, los eventos quedan en `offline_queue.db` y se reintentan solos.
- Cada 60 s el agente informa a la nube que está vivo (el panel muestra una alerta si deja de llegar).
- Registro en `logs/agente.log` (rota cada 5 MB). Las claves nunca se escriben en el registro.
