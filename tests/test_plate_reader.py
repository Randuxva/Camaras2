"""Fusión multi-cuadro de lecturas de patente (sin modelos: solo la lógica de voto)."""
import unittest

import numpy as np

from core.plate_reader import PlateRead, fuse_reads, normalize_plate, plate_format


def read(text, probs=None, idx=0):
    probs = probs if probs is not None else [0.95] * len(text)
    return PlateRead(text=text, char_probs=probs, det_conf=0.9, box=(0, 0, 100, 30), width_px=100,
                     sharpness=200.0, plate_img=np.zeros((30, 100, 3), np.uint8), source_index=idx)


class TestFormato(unittest.TestCase):
    def test_formatos_argentinos(self):
        self.assertEqual(plate_format("AF512KC"), "AUTO_MERCOSUR")
        self.assertEqual(plate_format("KPQ271"), "AUTO_1995")
        self.assertEqual(plate_format("A123BCD"), "MOTO_MERCOSUR")
        self.assertEqual(plate_format("123ABC"), "MOTO_2010")
        self.assertIsNone(plate_format("DAA43B"))
        self.assertEqual(normalize_plate("af 512-kc"), "AF512KC")


class TestFusion(unittest.TestCase):
    def test_tres_cuadros_iguales_es_fuerte(self):
        f = fuse_reads([read("AF512KC", idx=i) for i in range(3)])
        self.assertEqual(f.plate, "AF512KC")
        self.assertEqual(f.support, 3)
        self.assertTrue(f.is_strong())

    def test_voto_por_caracter_corrige_un_error_por_cuadro(self):
        # Caso real (Kangoo): ningún cuadro leyó todo bien, el voto sí
        a = read("AD417RW", [0.99, 0.99, 0.99, 0.99, 0.99, 0.95, 0.30])
        b = read("AD417HH", [0.99, 0.99, 0.99, 0.99, 0.99, 0.40, 0.97])
        c = read("AD417RH", [0.90, 0.90, 0.90, 0.90, 0.90, 0.80, 0.80])
        f = fuse_reads([a, b, c])
        self.assertEqual(f.plate, "AD417RH")

    def test_una_sola_lectura_no_es_fuerte(self):
        f = fuse_reads([read("AB274PL", [0.99] * 7)])
        self.assertEqual(f.plate, "AB274PL")
        self.assertFalse(f.is_strong())

    def test_lecturas_sin_formato_valido_no_votan(self):
        f = fuse_reads([read("2841C"), read("W"), read("")])
        self.assertIsNone(f.plate)
        self.assertFalse(f.is_strong())
        self.assertEqual(f.n_reads, 0)

    def test_desacuerdo_baja_la_confianza_del_caracter(self):
        f = fuse_reads([read("AA531TN"), read("AA501TN")])
        self.assertLess(f.min_char_conf, 0.75)
        self.assertFalse(f.is_strong())

    def test_payload(self):
        p = fuse_reads([read("KPQ271"), read("KPQ271")]).to_payload()
        self.assertEqual(p["plate"], "KPQ271")
        self.assertTrue(p["strong"])
        self.assertEqual(p["support"], 2)


if __name__ == "__main__":
    unittest.main()
