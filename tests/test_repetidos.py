"""Agrupar avisos repetidos: nunca se pierde un paso distinto de un mismo vehículo."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests._sin_yolo  # noqa: F401  (antes de importar el consumidor)
from core.repetidos import (AGRUPAR, ENVIAR_OTRO_SENTIDO, ENVIAR_SIN_CONFIRMAR, ENVIAR_PRIMERO, ENVIAR_SIN_PATENTE_FIRME, ENVIAR_SOMBRA,
                            ENVIAR_TOPE, RegistroRepetidos)
from core.roi_engine import GateEvent, ROIEvent
from pipeline.consumer import InferenceConsumer

P = "AB123CD"
T0 = 1_000_000.0


class RepetidosTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def reg(self, modo="true"):
        return RegistroRepetidos(str(self.dir / "q.db"), str(self.dir / "rep"), modo)

    resultado = ("CREATE_ENTRY", T0)          # cómo terminó en la nube el aviso anterior (por defecto: bien)

    def paso(self, r, t, direccion="ENTRADA", patente=P, firme=True, previo="BIEN"):
        res = (lambda *a: self.resultado) if previo == "BIEN" else (lambda *a: previo)
        dec, dt = r.decidir(patente=patente, firme=firme, direccion=direccion, ahora=T0 + t, resultado_previo=res)
        r.registrar(patente=patente, firme=firme, direccion=direccion, track_id=1, ruta="X", decision=dec,
                    desde_anterior=dt, crops=[np.zeros((10, 10, 3), np.uint8)], ahora=T0 + t)
        return dec

    def test_el_primer_aviso_siempre_se_envia(self):
        self.assertEqual(self.paso(self.reg(), 0), ENVIAR_PRIMERO)

    def test_el_mismo_sentido_en_menos_de_10_min_se_agrupa_si_el_anterior_termino_bien(self):
        r = self.reg()
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 60), AGRUPAR)
        self.assertEqual(self.paso(r, 590), AGRUPAR)

    def test_si_el_anterior_fue_a_revision_el_siguiente_se_envia_porque_puede_ser_el_que_se_registre(self):
        r = self.reg()
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 60, previo=("MANUAL_UNREADABLE", T0)), ENVIAR_SIN_CONFIRMAR)
        self.assertEqual(self.paso(r, 90, previo=("ERROR", T0)), ENVIAR_SIN_CONFIRMAR)

    def test_si_aun_no_hay_respuesta_del_anterior_se_envia(self):
        r = self.reg()
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 3, previo=None), ENVIAR_SIN_CONFIRMAR)

    def test_sin_funcion_de_consulta_se_envia_la_entrada_o_salida(self):
        r = self.reg()
        r.registrar(patente=P, firme=True, direccion="ENTRADA", track_id=1, ruta="X", decision=ENVIAR_PRIMERO, desde_anterior=None, ahora=T0)
        dec, _ = r.decidir(patente=P, firme=True, direccion="ENTRADA", ahora=T0 + 60)
        self.assertEqual(dec, ENVIAR_SIN_CONFIRMAR)

    def test_pasados_10_min_sin_avisos_es_un_aviso_nuevo(self):
        r = self.reg()
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 601), ENVIAR_PRIMERO)

    def test_una_entrada_una_salida_y_otra_entrada_son_tres_pasos(self):
        r = self.reg()
        self.assertEqual(self.paso(r, 0, "ENTRADA"), ENVIAR_PRIMERO)
        self.assertEqual(self.paso(r, 60, "SALIDA"), ENVIAR_OTRO_SENTIDO)
        self.assertEqual(self.paso(r, 120, "ENTRADA"), ENVIAR_OTRO_SENTIDO)

    def test_de_sin_direccion_a_una_direccion_o_al_reves_se_envia(self):
        r = self.reg()
        self.paso(r, 0, "ENTRADA")
        self.assertEqual(self.paso(r, 30, "DESCONOCIDA"), ENVIAR_OTRO_SENTIDO)      # podría ser la salida
        self.assertEqual(self.paso(r, 60, "DESCONOCIDA", previo=None), AGRUPAR)     # ahora sí, repetido (la nube los trata igual)
        self.assertEqual(self.paso(r, 90, "SALIDA"), ENVIAR_OTRO_SENTIDO)

    def test_sin_patente_firme_nunca_se_agrupa(self):
        r = self.reg()
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 10, firme=False), ENVIAR_SIN_PATENTE_FIRME)
        self.assertEqual(self.paso(r, 20, patente=None), ENVIAR_SIN_PATENTE_FIRME)

    def test_otra_patente_no_se_agrupa(self):
        r = self.reg()
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 10, patente="ZZ999ZZ"), ENVIAR_PRIMERO)

    def test_un_auto_que_se_queda_horas_deja_constancia_cada_hora(self):
        r = self.reg()
        self.paso(r, 0, "DESCONOCIDA")
        decisiones = [self.paso(r, t, "DESCONOCIDA", previo=None) for t in range(300, 3900, 300)]
        self.assertEqual(decisiones[:11], [AGRUPAR] * 11)
        self.assertEqual(self.paso(r, 3900, "DESCONOCIDA"), ENVIAR_TOPE)

    def test_modo_sombra_marca_pero_envia(self):
        r = self.reg("sombra")
        self.paso(r, 0)
        self.assertEqual(self.paso(r, 60), ENVIAR_SOMBRA)

    def test_lo_agrupado_queda_archivado_con_su_foto(self):
        r = self.reg()
        self.paso(r, 0)
        self.paso(r, 60)
        self.assertEqual(len(list((self.dir / "rep").glob("*.webp"))), 1)
        with r._conn() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM repetidos WHERE decision = ?", (AGRUPAR,)).fetchone()[0], 1)

    def test_la_memoria_sobrevive_a_un_reinicio(self):
        self.paso(self.reg(), time.time() - T0 - 30)
        r2 = self.reg()
        dec, _ = r2.decidir(patente=P, firme=True, direccion="DESCONOCIDA")
        self.assertEqual(dec, ENVIAR_OTRO_SENTIDO)          # el anterior era una ENTRADA
        dec, _ = r2.decidir(patente=P, firme=True, direccion="ENTRADA", resultado_previo=lambda *a: ("EXIT", T0))
        self.assertEqual(dec, AGRUPAR)


def _evento(direccion=GateEvent.ENTRADA):
    return ROIEvent(track_id=5, event_type=direccion, timestamp=1.0, centroid_y=0.7, bbox=(0, 0, 10, 10),
                    reason="CRUCE_COMPLETO", n_points=8)


def _fused(plate, fuerte=True):
    return SimpleNamespace(plate=plate, support=3, n_reads=3, min_char_conf=0.9,
                           is_strong=lambda: fuerte, to_payload=lambda: {"plate": plate})


class RepetidosConsumidorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _consumidor(self, modo="true"):
        c = InferenceConsumer.__new__(InferenceConsumer)
        c.quietos_descartados = 0
        c.quietos = None
        c.repetidos_agrupados = 0
        c.repetidos = RegistroRepetidos(str(self.dir / "q.db"), str(self.dir / "rep"), modo)
        c.plate_reader = MagicMock()
        c.plate_reader.read_many.return_value = []
        c.uploader = MagicMock()
        c.uploader.offline_queue.accion_previa.return_value = ("CREATE_ENTRY", 0)     # el anterior terminó bien
        c.enhancer = MagicMock()
        c.enhancer.enhance.side_effect = lambda im: (im, None)
        c._con_atributos = lambda meta, crop, tid: meta
        c._vehicle_class = lambda tid: "car"
        return c

    def _despachar(self, c, evento, fusion):
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", return_value=fusion):
            c._dispatch_with_plates(evento, raw, 1000.0, [None])

    def test_el_segundo_aviso_igual_no_se_envia_pero_queda_archivado(self):
        c = self._consumidor()
        self._despachar(c, _evento(), _fused(P))
        self._despachar(c, _evento(), _fused(P))
        self.assertEqual(c.uploader.save_and_dispatch.call_count, 1)
        self.assertEqual(c.repetidos_agrupados, 1)
        self.assertEqual(len(list((self.dir / "rep").glob("*.webp"))), 1)

    def test_si_el_aviso_anterior_fue_a_revision_el_siguiente_se_envia(self):
        c = self._consumidor()
        c.uploader.offline_queue.accion_previa.return_value = ("MANUAL_UNREADABLE", 0)
        self._despachar(c, _evento(), _fused(P))
        self._despachar(c, _evento(), _fused(P))
        self.assertEqual(c.uploader.save_and_dispatch.call_count, 2)

    def test_la_salida_tras_la_entrada_se_envia(self):
        c = self._consumidor()
        self._despachar(c, _evento(GateEvent.ENTRADA), _fused(P))
        self._despachar(c, _evento(GateEvent.SALIDA), _fused(P))
        self.assertEqual(c.uploader.save_and_dispatch.call_count, 2)

    def test_sin_patente_firme_se_envian_todos(self):
        c = self._consumidor()
        for _ in range(3):
            self._despachar(c, _evento(), _fused(P, fuerte=False))
        self.assertEqual(c.uploader.save_and_dispatch.call_count, 3)

    def test_modo_sombra_envia_todos(self):
        c = self._consumidor("sombra")
        for _ in range(3):
            self._despachar(c, _evento(), _fused(P))
        self.assertEqual(c.uploader.save_and_dispatch.call_count, 3)
        self.assertEqual(c.repetidos_agrupados, 0)

    def test_la_consulta_real_a_la_cola_local_devuelve_el_resultado_del_aviso_anterior(self):
        from cloud.offline_queue import OfflineQueue
        q = OfflineQueue(str(self.dir / "cola.db"))
        img = self.dir / "a.webp"
        img.write_bytes(b"x")
        t = int(time.time()) - 120
        eid = q.enqueue(track_id=1, event_type="SALIDA", image_path=str(img), timestamp=t, camera_id="cam",
                        sucursal_id="suc", local_ocr={"plate": P, "support": 3})
        self.assertIsNone(q.accion_previa(P, "SALIDA", t - 5, t + 5))                      # aún sin respuesta de la nube
        q.mark_completed(eid, {"success": True, "action": "EXIT", "data": {"movimiento": {"codigo": 7}}})
        self.assertEqual(q.accion_previa(P, "SALIDA", t - 5, t + 5), ("EXIT", t))
        self.assertIsNone(q.accion_previa(P, "ENTRADA", t - 5, t + 5))                     # otro sentido
        self.assertIsNone(q.accion_previa("ZZ999ZZ", "SALIDA", t - 5, t + 5))              # otra patente

    def test_apagado_envia_todos(self):
        c = self._consumidor()
        c.repetidos = None
        for _ in range(3):
            self._despachar(c, _evento(), _fused(P))
        self.assertEqual(c.uploader.save_and_dispatch.call_count, 3)


if __name__ == "__main__":
    unittest.main()
