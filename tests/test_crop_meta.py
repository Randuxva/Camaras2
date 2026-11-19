"""La calidad del recorte viaja con el evento: la nube la usa para elegir la foto de referencia de los abonados
y para saber si el color de la foto sirve."""
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from cloud.offline_queue import OfflineQueue
from core.frame_selector import FrameBuffer
import tests._sin_yolo  # noqa: F401  (antes de importar el consumidor)
from pipeline.consumer import InferenceConsumer

META = {"score": 1234.5, "rejilla": False, "truncated": False, "lighting": "NORMAL", "sinColor": False, "ancho": 600, "alto": 400}


def _auto(w=600, h=400, color=(30, 30, 200)):
    """Un 'auto' liso con algo de textura y color (rojo)."""
    img = np.full((h, w, 3), color, np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (200, 200, 200), 6)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


class CropMetaEnLaCola(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.img = self.tmp / "a.webp"
        self.img.write_bytes(b"x")

    def _enq(self, q, **kw):
        return q.enqueue(track_id=1, event_type="ENTRADA", image_path=str(self.img), timestamp=int(time.time()),
                         camera_id="cam", sucursal_id="suc", **kw)

    def test_viaja_y_vuelve_igual(self):
        q = OfflineQueue(str(self.tmp / "q.db"))
        self._enq(q, crop_meta=META)
        ev = q.get_pending(limit=10, due_only=False)[0]
        self.assertEqual(ev.crop_meta, META)

    def test_un_evento_sin_calidad_sigue_funcionando(self):
        q = OfflineQueue(str(self.tmp / "q.db"))
        self._enq(q)
        self.assertIsNone(q.get_pending(limit=10, due_only=False)[0].crop_meta)

    def test_una_cola_vieja_sin_la_columna_se_migra_sola_y_conserva_lo_pendiente(self):
        viejo = self.tmp / "viejo.db"
        conn = sqlite3.connect(viejo)
        conn.execute("""CREATE TABLE pending_events (id TEXT PRIMARY KEY, track_id INTEGER, event_type TEXT, image_path TEXT,
                        timestamp INTEGER, attempts INTEGER DEFAULT 0, created_at REAL, camera_id TEXT, sucursal_id TEXT,
                        status TEXT DEFAULT 'PENDING')""")
        conn.execute("INSERT INTO pending_events (id, track_id, event_type, image_path, timestamp, created_at, camera_id, sucursal_id) "
                     "VALUES ('e1', 7, 'SALIDA', ?, 1, ?, 'cam', 'suc')", (str(self.img), time.time()))
        conn.commit()
        conn.close()
        q = OfflineQueue(str(viejo))      # al abrirla agrega las columnas que faltan
        ev = q.get_pending(limit=10, due_only=False)[0]
        self.assertEqual((ev.id, ev.track_id, ev.event_type), ("e1", 7, "SALIDA"))
        self.assertIsNone(ev.crop_meta)


class CalidadDelRecorte(unittest.TestCase):
    def _candidato(self, img, **kw):
        buf = FrameBuffer(max_size=4, min_blur=0)
        buf.add(1, None, img, (0, 0, img.shape[1], img.shape[0]), **kw)
        return buf.get_best(1)

    def test_un_recorte_normal(self):
        m = InferenceConsumer._meta_de(self._candidato(_auto()))
        self.assertFalse(m["rejilla"])
        self.assertFalse(m["truncated"])
        self.assertFalse(m["sinColor"])
        self.assertGreater(m["score"], 0)
        self.assertEqual((m["ancho"], m["alto"]), (600, 400))

    def test_en_infrarrojo_el_color_no_sirve(self):
        gris = cv2.cvtColor(cv2.cvtColor(_auto(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        self.assertTrue(InferenceConsumer._meta_de(self._candidato(gris))["sinColor"])

    def test_la_reja_queda_marcada(self):
        img = cv2.GaussianBlur(_auto(), (0, 0), 25)
        for x in range(0, 600, 12):
            cv2.line(img, (x, 0), (x, 400), (10, 10, 10), 2)
        for y in range(0, 400, 12):
            cv2.line(img, (0, y), (600, y), (10, 10, 10), 2)
        self.assertTrue(InferenceConsumer._meta_de(self._candidato(img))["rejilla"])

    def test_vehiculo_cortado_por_el_borde(self):
        img = _auto()
        buf = FrameBuffer(max_size=4, min_blur=0)
        buf.add(1, None, img, (100, 100, 700, 500), frame_shape=(500, 2880))   # toca el borde inferior del cuadro
        self.assertTrue(InferenceConsumer._meta_de(buf.get_best(1))["truncated"])


if __name__ == "__main__":
    unittest.main()
