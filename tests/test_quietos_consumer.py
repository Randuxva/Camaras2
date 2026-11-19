"""El consumidor aplica las salvaguardas: un objeto quieto solo deja de enviarse si cumple TODAS las condiciones."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests._sin_yolo  # noqa: F401  (antes de importar el consumidor)
from core.quietos import RegistroQuietos
from core.roi_engine import GateEvent, ROIEvent
from pipeline.consumer import InferenceConsumer

SUV = (0.045, 0.356, 0.142, 0.606)
OTRO = (0.40, 0.40, 0.55, 0.70)


def _evento(sin_movimiento=True, caja=SUV):
    return ROIEvent(track_id=5, event_type=GateEvent.DESCONOCIDA, timestamp=1.0, centroid_y=0.6,
                    bbox=(0, 0, 10, 10), reason="TRACK_PERDIDO_SIN_DIRECCION", n_points=12,
                    sin_movimiento=sin_movimiento, norm_box=caja)


def _fused(plate):
    return SimpleNamespace(plate=plate, support=3, n_reads=3, min_char_conf=0.9,
                           is_strong=lambda: True, to_payload=lambda: {"plate": plate})


class QuietosConsumidorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _consumidor(self, modo="true", repeticiones_previas=0, clase="car"):
        reg = RegistroQuietos(str(self.dir / "q.db"), str(self.dir / "fotos"), modo)
        for i in range(repeticiones_previas):
            reg.registrar(track_id=i, tipo="DESCONOCIDA", ruta="X", n_puntos=5, caja=SUV, clase="car", con_patente=False,
                          segunda_lectura=False, repeticiones=0, decision="ENVIAR_LUGAR_NUEVO", ahora=time.time() - i)
        c = InferenceConsumer.__new__(InferenceConsumer)
        c.quietos_descartados = 0
        c.quietos = reg
        c.repetidos = None
        c.repetidos_agrupados = 0
        c.plate_reader = MagicMock()
        c.plate_reader.read_many.return_value = []
        c.uploader = MagicMock()
        c.enhancer = MagicMock()
        c.enhancer.enhance.side_effect = lambda im: (im, None)
        c._con_atributos = lambda meta, crop, tid: meta
        c._vehicle_class = lambda tid: clase
        return c

    def _despachar(self, c, evento, *fusiones):
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", side_effect=list(fusiones)):
            c._dispatch_with_plates(evento, raw, 1000.0, [None])

    def test_quieto_sin_patente_en_lugar_nuevo_se_envia(self):
        c = self._consumidor()
        self._despachar(c, _evento(), _fused(None))
        c.uploader.save_and_dispatch.assert_called_once()
        self.assertEqual(c.quietos_descartados, 0)

    def test_quieto_sin_patente_en_lugar_recurrente_no_se_envia_pero_queda_archivado(self):
        c = self._consumidor(repeticiones_previas=8)
        c.plate_reader.read_many.return_value = []
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", return_value=_fused(None)):
            c._dispatch_with_plates(_evento(), raw, 1000.0, [None])
        c.uploader.save_and_dispatch.assert_not_called()
        self.assertEqual(c.quietos_descartados, 1)
        self.assertEqual(len(list((self.dir / "fotos").glob("*.webp"))), 1)      # la foto no se pierde

    def test_en_otro_lugar_se_envia_aunque_el_recurrente_exista(self):
        c = self._consumidor(repeticiones_previas=8)
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", return_value=_fused(None)):
            c._dispatch_with_plates(_evento(caja=OTRO), raw, 1000.0, [None])
        c.uploader.save_and_dispatch.assert_called_once()

    def test_quieto_con_patente_se_envia_aunque_el_lugar_sea_recurrente(self):
        c = self._consumidor(repeticiones_previas=8)
        self._despachar(c, _evento(), _fused("AB123CD"))
        c.uploader.save_and_dispatch.assert_called_once()

    def test_segunda_lectura_con_la_imagen_realzada_puede_encontrar_la_patente(self):
        c = self._consumidor(repeticiones_previas=8)
        lectura = SimpleNamespace(source_index=0, quality=1.0, plate_img=np.zeros((10, 20, 3), np.uint8))
        c.plate_reader.read_many.side_effect = [[], [lectura]]
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", side_effect=[_fused(None), _fused("AB123CD")]), \
                patch("pipeline.consumer.upscale_plate", side_effect=lambda x: x):
            c._dispatch_with_plates(_evento(), raw, 1000.0, [None])
        c.uploader.save_and_dispatch.assert_called_once()
        self.assertEqual(c.plate_reader.read_many.call_count, 2)

    def test_una_moto_quieta_sin_patente_se_envia(self):
        c = self._consumidor(repeticiones_previas=8, clase="motorcycle")
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", return_value=_fused(None)):
            c._dispatch_with_plates(_evento(), raw, 1000.0, [None])
        c.uploader.save_and_dispatch.assert_called_once()

    def test_modo_sombra_envia_todo(self):
        c = self._consumidor(modo="sombra", repeticiones_previas=8)
        raw = [np.zeros((40, 60, 3), np.uint8)]
        with patch("pipeline.consumer.fuse_reads", return_value=_fused(None)):
            c._dispatch_with_plates(_evento(), raw, 1000.0, [None])
        c.uploader.save_and_dispatch.assert_called_once()
        self.assertEqual(c.quietos_descartados, 0)

    def test_un_evento_normal_sin_patente_se_envia_como_siempre(self):
        c = self._consumidor(repeticiones_previas=8)
        self._despachar(c, _evento(sin_movimiento=False), _fused(None))
        c.uploader.save_and_dispatch.assert_called_once()

    def test_sin_registro_de_quietos_se_envia_todo(self):
        c = self._consumidor()
        c.quietos = None
        self._despachar(c, _evento(), _fused(None))
        c.uploader.save_and_dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class ReglaEstrictaConsumidorTest(QuietosConsumidorTest):
    """ROI_SOLO_CRUCES: lo que no cruzó la línea no se envía (ni con patente, ni moto), pero queda archivado."""

    def _estricto(self, modo="true", clase="car"):
        c = self._consumidor(clase=clase)
        c.quietos = RegistroQuietos(str(self.dir / "e.db"), str(self.dir / "fotos_e"), modo, estricto=True)
        return c

    def test_sin_cruce_con_patente_no_se_envia_y_queda_archivado(self):
        c = self._estricto()
        self._despachar(c, _evento(), _fused("AB123CD"))
        c.uploader.save_and_dispatch.assert_not_called()
        self.assertEqual(len(list((self.dir / "fotos_e").glob("*.webp"))), 1)
        with c.quietos._conn() as con:
            self.assertEqual(con.execute("SELECT decision FROM quietos").fetchone()[0], "ARCHIVADO_SIN_CRUCE")

    def test_sin_cruce_moto_tampoco_se_envia(self):
        c = self._estricto(clase="motorcycle")
        self._despachar(c, _evento(), _fused(None), _fused(None))
        c.uploader.save_and_dispatch.assert_not_called()

    def test_cruce_completo_se_envia_normal(self):
        c = self._estricto()
        self._despachar(c, _evento(sin_movimiento=False), _fused("AB123CD"))
        c.uploader.save_and_dispatch.assert_called_once()

    def test_modo_sombra_estricto_envia_todo_y_lo_marca(self):
        c = self._estricto(modo="sombra")
        self._despachar(c, _evento(), _fused(None), _fused(None))
        c.uploader.save_and_dispatch.assert_called_once()
        with c.quietos._conn() as con:
            self.assertEqual(con.execute("SELECT decision FROM quietos").fetchone()[0], "ENVIAR_SOMBRA_SIN_CRUCE")
