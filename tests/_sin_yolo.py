"""Permite importar pipeline.consumer sin ultralytics/torch instalados (pesan GB y el CI no los instala).

Las pruebas que lo usan no cargan el modelo YOLO: solo prueban la lógica del consumidor. Si ultralytics
está instalado (la PC del portón), se usa el real.
"""
import sys
import types

try:
    import ultralytics  # noqa: F401
except ImportError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=None)
