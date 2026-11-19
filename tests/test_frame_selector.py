"""Pruebas del buffer de mejores recortes."""
import unittest

import cv2
import numpy as np

from core.frame_selector import FrameBuffer, es_rejilla


def textured(w, h, blur=0):
    rng = np.random.default_rng(w * h)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), blur) if blur else img


class FrameBufferTest(unittest.TestCase):
    def test_conserva_los_mejores_y_no_los_mas_nuevos(self):
        buf = FrameBuffer(max_size=2, min_blur=0)
        sharp_big = textured(600, 400, blur=1)  # blur=1: el ruido puro tiene bordes por todos lados, como una reja
        buf.add(1, None, sharp_big, (0, 0, 600, 400))
        for _ in range(5):  # frames posteriores peores (borrosos, chicos)
            buf.add(1, None, textured(200, 120, blur=3), (0, 0, 200, 120))
        best = buf.get_best(1)
        self.assertEqual(best.crop.shape[:2], (400, 600))
        self.assertEqual(len(buf.get_top(1, 3)), 2)

    def test_prefiere_vehiculo_completo_a_primer_plano_cortado(self):
        # Caso validación #86: el primer plano lateral junto a la cámara (cortado, enorme)
        # no debe ganarle a la vista frontal completa en el portón.
        buf = FrameBuffer(max_size=3, min_blur=0)
        frente = textured(550, 430, blur=1)
        buf.add(1, None, frente, (1000, 800, 1550, 1230), frame_shape=(1620, 2880))
        primer_plano = textured(1400, 650)
        buf.add(1, None, primer_plano, (1480, 970, 2880, 1620), frame_shape=(1620, 2880))
        best = buf.get_best(1)
        self.assertFalse(best.truncated)
        self.assertEqual(best.crop.shape[:2], (430, 550))
        self.assertEqual(len(buf.get_top(1, 3)), 2)

    def test_acepta_un_recorte_borroso_si_no_hay_otro(self):
        buf = FrameBuffer(max_size=4, min_blur=10_000)
        buf.add(7, None, textured(300, 200, blur=4), (0, 0, 300, 200))
        self.assertIsNotNone(buf.get_best(7))
        buf.add(7, None, textured(300, 200, blur=4), (0, 0, 300, 200))
        self.assertEqual(len(buf.get_top(7, 5)), 1)  # el segundo borroso se descarta

    def test_no_guarda_referencia_al_frame(self):
        buf = FrameBuffer(max_size=2, min_blur=0)
        frame = textured(800, 600)
        crop = frame[100:400, 100:500]
        buf.add(3, frame, crop, (100, 100, 500, 400))
        cand = buf.get_best(3)
        self.assertFalse(np.shares_memory(cand.crop, frame))


def con_reja(w=600, h=400, paso=12, grosor=2):
    """Auto (fondo suave) visto a través de la malla del portón."""
    img = cv2.GaussianBlur(textured(w, h), (0, 0), 25)
    for x in range(0, w, paso):
        cv2.line(img, (x, 0), (x, h), (10, 10, 10), grosor)
    for y in range(0, h, paso):
        cv2.line(img, (0, y), (w, y), (10, 10, 10), grosor)
    return img


class RejillaTest(unittest.TestCase):
    def test_detecta_la_malla_y_no_un_cuadro_limpio(self):
        self.assertTrue(es_rejilla(con_reja()))
        self.assertTrue(es_rejilla(con_reja(paso=16, grosor=3)))
        self.assertFalse(es_rejilla(cv2.GaussianBlur(textured(600, 400), (0, 0), 25)))
        self.assertFalse(es_rejilla(textured(600, 400, blur=1)))
        self.assertFalse(es_rejilla(np.zeros((0, 0, 3), np.uint8)))

    def test_el_cuadro_limpio_le_gana_a_la_reja_aunque_sea_menos_nitido(self):
        # Antes la malla puntuaba más (más bordes = "más nítido") y se elegía como foto del vehículo
        buf = FrameBuffer(max_size=4, min_blur=0)
        buf.add(1, None, con_reja(), (0, 0, 600, 400))
        limpio = textured(600, 400, blur=1.5)
        buf.add(1, None, limpio, (0, 0, 600, 400))
        best = buf.get_best(1)
        self.assertFalse(best.rejilla)
        self.assertTrue(np.array_equal(best.crop, limpio))

    def test_con_el_buffer_lleno_se_descarta_la_reja_y_no_el_cuadro_limpio(self):
        buf = FrameBuffer(max_size=2, min_blur=0)
        limpio = textured(600, 400, blur=1.5)
        buf.add(1, None, limpio, (0, 0, 600, 400))
        for _ in range(4):
            buf.add(1, None, con_reja(), (0, 0, 600, 400))
        top = buf.get_top(1, 5)
        self.assertEqual(len(top), 2)
        self.assertFalse(top[0].rejilla)

    def test_si_todo_es_reja_igual_se_conserva_como_ultimo_recurso(self):
        buf = FrameBuffer(max_size=3, min_blur=0)
        buf.add(1, None, con_reja(), (0, 0, 600, 400))
        self.assertIsNotNone(buf.get_best(1))


if __name__ == "__main__":
    unittest.main()
