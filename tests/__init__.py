import os

# Las pruebas no escriben en el archivo de registro del agente
os.environ.setdefault("LOG_FILE", "")
