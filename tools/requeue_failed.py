"""
Vuelve a poner en cola los eventos que el agente había dado por perdidos (estado FAILED) para
que se suban con su hora original. El agente en ejecución los toma solo, del más viejo al más nuevo.

Usar SOLO con la Edge Function que respeta la hora del evento (2026-10-09 o posterior): la
anterior los habría registrado con la hora del reenvío.

    python tools/requeue_failed.py            # muestra qué haría
    python tools/requeue_failed.py --apply    # lo hace
    python tools/requeue_failed.py --apply --solo-con-patente   # recomendado para eventos de días atrás
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
os.chdir(AGENT)

from cloud.offline_queue import OfflineQueue  # noqa: E402
from config import cfg  # noqa: E402


def main() -> None:
    apply = "--apply" in sys.argv
    conn = sqlite3.connect(cfg.offline_queue_db_path)
    conn.row_factory = sqlite3.Row
    solo = "--solo-con-patente" in sys.argv
    rows = conn.execute("SELECT event_type, image_path, created_at, local_ocr FROM pending_events WHERE status = 'FAILED'").fetchall()
    conn.close()
    con_foto = [r for r in rows if os.path.exists(r["image_path"])]
    con_patente = [r for r in con_foto if ((json.loads(r["local_ocr"]) if r["local_ocr"] else None) or {}).get("plate")]
    print(f"Con patente leída en el portón: {len(con_patente)} | sin patente: {len(con_foto) - len(con_patente)}")
    if solo:
        con_foto = con_patente
    por_tipo = Counter(r["event_type"] for r in con_foto)
    print(f"Eventos FAILED: {len(rows)} | con foto (recuperables): {len(con_foto)} {dict(por_tipo)}")
    if con_foto:
        ts = sorted(r["created_at"] for r in con_foto)
        fmt = lambda t: time.strftime("%d/%m %H:%M", time.localtime(t))  # noqa: E731
        print(f"Del {fmt(ts[0])} al {fmt(ts[-1])}")
    if not apply:
        print("Nada cambió. Agregá --apply para reencolarlos.")
        return
    n = OfflineQueue(cfg.offline_queue_db_path).requeue_failed(solo_con_patente=solo)
    print(f"Reencolados: {n}. El agente los sube solo (5 cada {cfg.offline_retry_interval_sec:.0f} s).")


if __name__ == "__main__":
    main()
