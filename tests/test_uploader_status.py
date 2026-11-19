"""Pruebas del latido al panel y del secreto del agente."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import cloud.uploader as uploader_mod
from cloud.offline_queue import OfflineQueue
from config import cfg


class UploaderStatusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # SQLite retiene el archivo en Windows
        self.queue = OfflineQueue(db_path=str(Path(self.tmp.name) / "q.db"))
        # Sin Supabase ni reintentos en segundo plano: solo se prueba el armado del latido
        self.cfg = replace(
            cfg, supabase_url="", supabase_key="anon", webhook_url="https://example.test/fn",
            agent_secret="s3cr3t", agent_status_interval_sec=60, default_sucursal_id="suc-1", camera_id="cam-1",
            offline_retry_interval_sec=3600,
        )
        self.patch_cfg = mock.patch.object(uploader_mod, "cfg", self.cfg)
        self.patch_cfg.start()
        # El worker de fondo también manda latidos: sin esto competía con la prueba (fallaba a veces)
        with mock.patch.object(uploader_mod.CloudUploader, "_retry_worker_loop", lambda self: None):
            self.up = uploader_mod.CloudUploader(offline_queue=self.queue)

    def tearDown(self):
        self.up.stop()
        self.patch_cfg.stop()
        self.tmp.cleanup()

    def test_latido_incluye_estado_de_camara_y_secreto(self):
        self.up.status_provider = lambda: {"camera_ok": True, "last_frame_age_sec": 0.4}
        with mock.patch.object(uploader_mod.requests, "post") as post:
            post.return_value.ok = True
            self.up._report_status()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://example.test/fn")
        self.assertEqual(kwargs["headers"]["x-agent-secret"], "s3cr3t")
        body = kwargs["json"]
        self.assertEqual(body["type"], "heartbeat")
        self.assertEqual(body["sucursal_id"], "suc-1")
        self.assertTrue(body["camera_ok"])
        self.assertEqual(body["queue_pending"], 0)

    def test_respeta_el_intervalo(self):
        with mock.patch.object(uploader_mod.requests, "post") as post:
            post.return_value.ok = True
            self.up._report_status()
            self.up._report_status()
        self.assertEqual(post.call_count, 1)

    def test_un_fallo_de_red_no_propaga_la_excepcion(self):
        with mock.patch.object(uploader_mod.requests, "post", side_effect=OSError("sin red")):
            self.up._report_status()  # no debe lanzar


if __name__ == "__main__":
    unittest.main()
