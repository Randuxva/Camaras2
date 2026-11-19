"""Color y tipo medidos en el agente (sin IA en la nube): respaldo para cuando Gemini no responde."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from core.vehicle_attrs import atributos_de, color_de, familia_de, tipo_de
import tests._sin_yolo  # noqa: F401  (antes de importar el consumidor)
from pipeline.consumer import InferenceConsumer


def _auto(bgr, w=600, h=420):
    """Fondo gris de calle y un 'auto' de un solo color donde está la pintura (capó y puertas)."""
    img = np.full((h, w, 3), (110, 110, 110), np.uint8)
    cv2.rectangle(img, (int(w * .10), int(h * .25)), (int(w * .90), int(h * .90)), bgr, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


class ColorLocal(unittest.TestCase):
    def test_negro(self):
        r = color_de(_auto((25, 25, 25)))
        self.assertEqual((r.color, r.familia), ("NEGRO", "OSCURO"))
        self.assertGreaterEqual(r.confianza, 0.5)

    def test_rojo(self):
        r = color_de(_auto((30, 30, 200)))
        self.assertEqual((r.color, r.familia), ("ROJO", "COLOR"))
        self.assertGreaterEqual(r.confianza, 0.5)

    def test_azul_se_informa_pero_sin_confianza_para_guardarlo(self):
        r = color_de(_auto((200, 80, 30)))
        self.assertEqual(r.color, "AZUL")
        self.assertLess(r.confianza, 0.5)

    def test_claro_no_inventa_el_nombre(self):
        # Blanco, plata y gris no se separan: la familia es CLARO y la confianza queda por debajo de lo que se guarda
        for bgr in ((245, 245, 245), (190, 190, 190), (140, 140, 140)):
            r = color_de(_auto(bgr))
            self.assertEqual(r.familia, "CLARO", bgr)
            self.assertLess(r.confianza, 0.5, bgr)

    def test_recortes_inservibles(self):
        for malo in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((20, 20, 3), np.uint8), np.zeros((100, 100), np.uint8)):
            r = color_de(malo)
            self.assertIsNone(r.color)
            self.assertEqual(r.confianza, 0.0)

    def test_familias(self):
        self.assertEqual(familia_de("NEGRO"), "OSCURO")
        self.assertEqual(familia_de("plata"), "CLARO")
        self.assertEqual(familia_de("ROJO"), "COLOR")
        self.assertEqual(familia_de(None), "DESCONOCIDO")


class TipoLocal(unittest.TestCase):
    def test_clase_del_detector(self):
        self.assertEqual(tipo_de("car"), "AUTO")
        self.assertEqual(tipo_de("Motorcycle"), "MOTO")
        self.assertEqual(tipo_de("truck"), "CAMIONETA")
        self.assertEqual(tipo_de("bus"), "OTRO")
        self.assertIsNone(tipo_de(""))
        self.assertIsNone(tipo_de(None))
        self.assertIsNone(tipo_de("persona"))

    def test_atributos_completos(self):
        a = atributos_de(_auto((25, 25, 25)), "car")
        self.assertEqual(a["colorLocal"], "NEGRO")
        self.assertEqual(a["colorLocalFamilia"], "OSCURO")
        self.assertEqual(a["tipoLocal"], "AUTO")
        self.assertNotIn("tipoLocal", atributos_de(_auto((25, 25, 25)), None))


class EnElConsumidor(unittest.TestCase):
    def _consumidor(self, clase="car"):
        c = InferenceConsumer.__new__(InferenceConsumer)
        c._vehicle_class = lambda track_id: clase
        return c

    def test_se_suma_a_la_calidad_del_recorte_sin_perderla(self):
        meta = {"score": 10.0, "rejilla": False}
        out = self._consumidor()._con_atributos(meta, _auto((25, 25, 25)), 7)
        self.assertEqual(out["score"], 10.0)
        self.assertEqual(out["colorLocal"], "NEGRO")
        self.assertEqual(out["tipoLocal"], "AUTO")
        self.assertNotIn("colorLocal", meta)  # no pisa el original

    def test_sin_calidad_previa_igual_viaja(self):
        self.assertEqual(self._consumidor("motorcycle")._con_atributos(None, _auto((30, 30, 200)), 1)["tipoLocal"], "MOTO")

    def test_si_falla_el_envio_sigue_con_lo_que_habia(self):
        meta = {"score": 1.0}
        self.assertEqual(self._consumidor()._con_atributos(meta, "no es una imagen", 1), meta)


if __name__ == "__main__":
    unittest.main()
