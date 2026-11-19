"""Salvaguardas de los objetos quietos: nunca se descarta un vehículo por no tener patente."""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from core.quietos import (ENVIAR_MOTO, ENVIAR_NORMAL, ENVIAR_NUEVO, ENVIAR_PATENTE, ENVIAR_SOMBRA, SUPRIMIR,
                          RegistroQuietos, iou, mismo_lugar, realzar)

SUV = (0.045, 0.356, 0.142, 0.606)          # el auto estacionado afuera, visto a través de la reja
OTRO_LUGAR = (0.40, 0.40, 0.55, 0.70)


class QuietosTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def reg(self, modo="true", **kw):
        return RegistroQuietos(str(self.dir / "q.db"), str(self.dir / "fotos"), modo, **kw)

    def ver(self, r, caja, n=1, con_patente=False, clase="car", t0=None):
        t0 = t0 if t0 is not None else time.time()
        for i in range(n):
            r.registrar(track_id=i, tipo="DESCONOCIDA", ruta="TRACK_PERDIDO_SIN_DIRECCION", n_puntos=10, caja=caja,
                        clase=clase, con_patente=con_patente, segunda_lectura=False, repeticiones=0,
                        decision=ENVIAR_NUEVO, crops=[np.zeros((20, 20, 3), np.uint8)], ahora=t0 + i)

    def decidir(self, r, caja=SUV, **kw):
        base = dict(sin_movimiento=True, con_patente=False, clase="car", caja=caja)
        base.update(kw)
        return r.decidir(**base)[0]

    def test_un_paso_normal_se_envia_siempre(self):
        self.assertEqual(self.decidir(self.reg(), sin_movimiento=False), ENVIAR_NORMAL)

    def test_quieto_con_patente_se_envia_aunque_el_lugar_sea_recurrente(self):
        r = self.reg()
        self.ver(r, SUV, 10)
        self.assertEqual(self.decidir(r, con_patente=True), ENVIAR_PATENTE)

    def test_una_moto_se_envia_siempre(self):
        r = self.reg()
        self.ver(r, SUV, 10)
        self.assertEqual(self.decidir(r, clase="motorcycle"), ENVIAR_MOTO)

    def test_los_primeros_avisos_de_un_lugar_siempre_se_envian(self):
        r = self.reg()
        for _ in range(5):
            self.assertEqual(self.decidir(r), ENVIAR_NUEVO)
            self.ver(r, SUV)
        self.assertEqual(self.decidir(r), SUPRIMIR)          # recién al sexto aviso igual, en el mismo lugar

    def test_con_cero_repeticiones_sin_cruce_no_hay_aviso_pero_patente_y_moto_si(self):
        r = self.reg("sombra", min_repeticiones=0)
        self.assertEqual(self.decidir(r), ENVIAR_SOMBRA)                      # primer aviso, sin repetición previa
        self.assertEqual(self.decidir(r, con_patente=True), ENVIAR_PATENTE)
        self.assertEqual(self.decidir(r, clase="motorcycle"), ENVIAR_MOTO)
        self.assertEqual(self.decidir(r, sin_movimiento=False), ENVIAR_NORMAL)
        self.assertEqual(self.decidir(self.reg("true", min_repeticiones=0)), SUPRIMIR)

    def test_un_lugar_nuevo_se_envia_aunque_haya_otro_recurrente(self):
        r = self.reg()
        self.ver(r, SUV, 20)
        self.assertEqual(self.decidir(r, caja=SUV), SUPRIMIR)
        self.assertEqual(self.decidir(r, caja=OTRO_LUGAR), ENVIAR_NUEVO)

    def test_sin_caja_no_se_suprime(self):
        r = self.reg()
        self.ver(r, SUV, 20)
        self.assertEqual(self.decidir(r, caja=None), ENVIAR_NUEVO)

    def test_modo_sombra_marca_pero_envia_todo(self):
        r = self.reg("sombra")
        self.ver(r, SUV, 20)
        self.assertEqual(self.decidir(r), ENVIAR_SOMBRA)

    def test_lo_de_hace_mas_de_24_horas_no_cuenta(self):
        r = self.reg()
        self.ver(r, SUV, 20, t0=time.time() - 30 * 3600)
        self.assertEqual(self.decidir(r), ENVIAR_NUEVO)

    def test_los_avisos_con_patente_no_ensenan_lugares_recurrentes(self):
        r = self.reg()
        self.ver(r, SUV, 20, con_patente=True)
        self.assertEqual(self.decidir(r), ENVIAR_NUEVO)

    def test_el_historial_sobrevive_a_un_reinicio(self):
        self.ver(self.reg(), SUV, 8)
        self.assertEqual(self.decidir(self.reg()), SUPRIMIR)

    def test_lo_suprimido_queda_archivado_y_nunca_se_borra(self):
        r = self.reg()
        r.registrar(track_id=7, tipo="DESCONOCIDA", ruta="X", n_puntos=3, caja=SUV, clase="car", con_patente=False,
                    segunda_lectura=False, repeticiones=9, decision=SUPRIMIR, crops=[np.full((30, 30, 3), 90, np.uint8)])
        fotos = list((self.dir / "fotos").glob("*.webp"))
        self.assertEqual(len(fotos), 1)
        with r._conn() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM quietos WHERE decision = ?", (SUPRIMIR,)).fetchone()[0], 1)
        self.assertEqual(r.suprimidos, 1)

    def test_el_mismo_auto_visto_entero_o_como_franja_es_el_mismo_lugar(self):
        franja = (0.06, 0.36, 0.10, 0.60)             # una franja angosta dentro de la caja entera del SUV
        self.assertTrue(mismo_lugar(SUV, franja))
        r = self.reg()
        self.ver(r, SUV, 10)
        self.assertEqual(self.decidir(r, caja=franja), SUPRIMIR)
        self.assertFalse(mismo_lugar(SUV, OTRO_LUGAR))
        self.assertEqual(self.decidir(r, caja=OTRO_LUGAR), ENVIAR_NUEVO)

    def test_iou_y_realce(self):
        self.assertAlmostEqual(iou(SUV, SUV), 1.0)
        self.assertEqual(iou(SUV, OTRO_LUGAR), 0.0)
        self.assertEqual(realzar(np.zeros((20, 30, 3), np.uint8)).shape, (40, 60, 3))


if __name__ == "__main__":
    unittest.main()


class ReglaEstrictaQuietosTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_estricta_archiva_todo_lo_que_no_cruzo_sin_excepciones(self):
        from core.quietos import ARCHIVAR_SIN_CRUCE, SOMBRA_SIN_CRUCE
        r = RegistroQuietos(str(self.dir / "q.db"), str(self.dir / "f"), "true", estricto=True)
        for kw in ({}, {"con_patente": True}, {"clase": "motorcycle"}):
            base = dict(sin_movimiento=True, con_patente=False, clase="car", caja=SUV); base.update(kw)
            self.assertEqual(r.decidir(**base)[0], ARCHIVAR_SIN_CRUCE)
        self.assertEqual(r.decidir(sin_movimiento=False, con_patente=False, clase="car", caja=SUV)[0], ENVIAR_NORMAL)
        s = RegistroQuietos(str(self.dir / "q2.db"), str(self.dir / "f2"), "sombra", estricto=True)
        self.assertEqual(s.decidir(sin_movimiento=True, con_patente=False, clase="car", caja=SUV)[0], SOMBRA_SIN_CRUCE)
