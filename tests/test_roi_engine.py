"""
Pruebas del motor de dirección por trayectoria.
Los casos reproducen situaciones reales de la auditoría del 2026-09-28.
Ejecutar desde anpr-edge/:  python -m unittest discover -s tests -t .
"""
import unittest

from core.roi_engine import GateEvent, ROIEngine

H, W = 1620, 2880
LINE_B, LINE_A = 0.56, 0.68


def bbox(y_bottom: float, x_center: float = 0.4, half_w: float = 0.1, height: float = 0.25):
    y2 = int(y_bottom * H)
    y1 = max(0, int((y_bottom - height) * H))
    x1 = int((x_center - half_w) * W)
    x2 = int((x_center + half_w) * W)
    return (x1, y1, x2, y2)


class ROIEngineTest(unittest.TestCase):
    def setUp(self):
        self.eng = ROIEngine(line_a_y_ratio=LINE_A, line_b_y_ratio=LINE_B,
                             track_lost_sec=2.0, min_displacement_ratio=0.04)

    def feed(self, track_id, ys, t0=0.0, dt=0.3, xs=None):
        events = []
        for i, y in enumerate(ys):
            x = xs[i] if xs else 0.4
            ev = self.eng.update(track_id, bbox(y, x), H, W, timestamp=t0 + i * dt)
            if ev:
                events.append(ev)
        return events

    def finish(self, at):
        return self.eng.collect_finished(at)

    def test_entrada_cruce_completo(self):
        # decisión en t=1.2 (y=0.75); se retiene 1,5 s para juntar fotos más cercanas
        self.assertEqual(self.feed(1, [0.45, 0.52, 0.60, 0.66, 0.75]), [])
        self.assertEqual(self.finish(2.5), [])
        evs = self.finish(2.8)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].event_type, GateEvent.ENTRADA)
        self.assertEqual(evs[0].reason, "CRUCE_COMPLETO")
        self.assertEqual(self.finish(100), [])  # no se emite dos veces

    def test_evento_retenido_incluye_puntos_posteriores(self):
        self.feed(12, [0.45, 0.60, 0.70], dt=0.3)       # decisión en t=0.6
        self.feed(12, [0.80], t0=1.0)
        evs = self.finish(2.2)                           # sigue visible, pasó la retención
        self.assertEqual([e.event_type for e in evs], [GateEvent.ENTRADA])
        self.assertEqual(evs[0].n_points, 4)
        self.feed(12, [0.90], t0=2.3)
        self.assertEqual(self.finish(10), [])

    def test_salida_cruce_completo(self):
        self.feed(2, [0.95, 0.85, 0.70, 0.60, 0.50])
        self.assertEqual([e.event_type for e in self.finish(10)], [GateEvent.SALIDA])

    def test_salida_vista_una_sola_vez_abajo(self):
        # Caso Toyota Etios: detectado una única vez en y=0.92 (ya adentro) → SALIDA
        self.feed(3, [0.92])
        evs = self.finish(10)
        self.assertEqual([e.event_type for e in evs], [GateEvent.SALIDA])

    def test_entrada_con_pocos_frames_por_desplazamiento(self):
        # Caso Fiat rojo: visto en el portón y luego más cerca, sin llegar a la línea A
        self.feed(4, [0.577, 0.64])
        evs = self.finish(10)
        self.assertEqual([e.event_type for e in evs], [GateEvent.ENTRADA])
        self.assertEqual(evs[0].reason, "TRACK_PERDIDO_DESPLAZAMIENTO")

    def test_salida_con_pocos_frames_por_desplazamiento(self):
        # Caso sedán blanco / Spark rojo: primera detección en el portal alejándose
        self.feed(5, [0.658, 0.60])
        evs = self.finish(10)
        self.assertEqual([e.event_type for e in evs], [GateEvent.SALIDA])

    def test_visto_una_vez_en_portal_es_desconocida(self):
        self.feed(6, [0.60])
        self.assertEqual(self.finish(10), [])        # DESCONOCIDA se retiene 3 s (posible fragmento)
        evs = self.finish(14)
        self.assertEqual([e.event_type for e in evs], [GateEvent.DESCONOCIDA])

    def test_una_deteccion_rozando_linea_b_se_ignora(self):
        # Caso validación 19:26: visto una vez en y=0.562 (auto de la calle junto al cordón)
        self.feed(70, [0.562])
        self.assertEqual(self.finish(10), [])

    def test_trafico_de_calle_ignorado(self):
        # Colectivo / camioneta pasando por la calle: nunca baja de la línea B
        self.feed(7, [0.48, 0.49, 0.487, 0.49], xs=[0.1, 0.3, 0.5, 0.7])
        self.assertEqual(self.finish(10), [])

    def test_auto_estacionado_enfrente_ignorado(self):
        self.feed(8, [0.49] * 50, dt=1.0)
        self.assertEqual(self.finish(100), [])

    def test_paso_lateral_por_vereda_ignorado(self):
        # Auto maniobrando en el cordón: se mueve de costado en el portal sin entrar
        self.feed(9, [0.58, 0.585, 0.58, 0.582], xs=[0.1, 0.3, 0.5, 0.7])
        self.assertEqual(self.finish(10), [])

    def test_track_activo_no_se_cierra_antes_de_tiempo(self):
        self.feed(10, [0.60, 0.62], t0=0.0)
        self.assertEqual(self.finish(1.5), [])      # visto hace <2 s
        self.assertEqual(self.eng.active_track_count, 1)
        self.assertEqual(self.finish(5.0), [])        # perdido: DESCONOCIDA retenida 3 s
        self.assertEqual(self.eng.active_track_count, 0)
        self.assertEqual(len(self.finish(8.5)), 1)

    def test_is_candidate_solo_en_portal(self):
        self.feed(11, [0.45])
        self.assertFalse(self.eng.is_candidate(11))
        self.feed(11, [0.60], t0=1.0)
        self.assertTrue(self.eng.is_candidate(11))

    def test_cambio_de_id_tras_entrada_no_genera_salida(self):
        # Caso validación 18:31: auto blanco entra (CRUCE_COMPLETO) y al pasar junto a la
        # cámara el tracker le da un ID nuevo visto una vez en y=0.999 → no debe ser SALIDA.
        self.feed(20, [0.487, 0.52, 0.56, 0.60, 0.70], xs=[0.6] * 5, dt=0.8)
        self.eng.update(21, bbox(0.999, x_center=0.75), H, W, timestamp=4.0)
        self.assertEqual([e.event_type for e in self.finish(20)], [GateEvent.ENTRADA])
        self.assertEqual(self.eng.total_continuations, 1)

    def test_auto_maniobrando_junto_a_la_camara_no_genera_salidas(self):
        # Caso validación 18:52: el mismo auto reaparece 3 veces junto a la cámara durante ~9 s
        self.feed(60, [0.49, 0.56, 0.64, 0.70], dt=0.5)
        self.assertEqual([e.event_type for e in self.finish(3.5)], [GateEvent.ENTRADA])
        for i, t in enumerate((5.0, 9.0, 13.0, 17.0)):
            self.eng.update(61 + i, bbox(0.84 + i * 0.04, x_center=0.45), H, W, timestamp=t)
            self.assertEqual(self.finish(t + 2.5), [])
        self.assertEqual(self.eng.total_continuations, 4)

    def test_otro_auto_que_entra_despues_de_una_salida_no_se_suprime(self):
        self.feed(30, [0.95, 0.85, 0.70, 0.60, 0.50])            # SALIDA
        self.feed(31, [0.45, 0.55, 0.62, 0.72], t0=3.0)            # otro auto entra desde la calle
        kinds = sorted(e.event_type.value for e in self.finish(20))
        self.assertEqual(kinds, ["ENTRADA", "SALIDA"])

    def test_continuacion_de_salida_por_el_portal(self):
        self.feed(40, [0.999])                                     # visto una vez junto a la cámara
        self.assertEqual([e.event_type for e in self.finish(3.0)], [GateEvent.SALIDA])
        self.feed(41, [0.66, 0.60], t0=3.5)                        # nuevo ID alejándose por el portal
        self.assertEqual(self.finish(10), [])

    def test_auto_que_sale_despues_de_una_entrada_lejos_en_el_tiempo(self):
        self.feed(50, [0.45, 0.60, 0.72])                           # ENTRADA
        self.assertEqual([e.event_type for e in self.finish(5)], [GateEvent.ENTRADA])
        self.feed(51, [0.99, 0.80, 0.62], t0=30.0)                  # otro auto sale 30 s después
        self.assertEqual([e.event_type for e in self.finish(40)], [GateEvent.SALIDA])

    def test_poligono_roi(self):
        eng = ROIEngine(line_a_y_ratio=LINE_A, line_b_y_ratio=LINE_B,
                        polygon_ratios="0,0,0.6,0,0.6,1,0,1")
        self.assertIsNone(eng.update(1, bbox(0.9, x_center=0.9, half_w=0.05), H, W, timestamp=0))
        self.assertEqual(eng.active_track_count, 0)  # fuera del polígono: ignorado
        eng.update(2, bbox(0.9, x_center=0.3), H, W, timestamp=0)
        self.assertEqual(eng.active_track_count, 1)


    # ── Grabación del 2026-10-02 (pipeline de dos streams) ──────────────────────────────
    def test_auto_estacionado_junto_al_porton_no_mantiene_yolo(self):
        # quieto en y≈0,48 (por encima de B - 0,08): antes contaba como "cerca" y YOLO no paraba
        self.feed(47, [0.483, 0.489, 0.486, 0.489, 0.483] * 4, dt=0.5)
        self.assertEqual(self.eng.relevant_track_count(), 0)

    def test_auto_acercandose_si_cuenta(self):
        self.feed(5, [0.44, 0.46, 0.48, 0.50, 0.52])
        self.assertEqual(self.eng.relevant_track_count(), 1)

    def test_moto_partida_en_dos_tracks_es_un_solo_evento(self):
        # Orden real (grabación 2026-10-02): el fragmento (46, visto una vez en el portal) se pierde
        # ANTES que el track principal (2), que termina en ENTRADA por desplazamiento.
        self.feed(2, [0.492], t0=0.0)
        self.feed(46, [0.586], t0=2.0)
        self.feed(2, [0.567], t0=3.0)
        first = self.finish(4.5)                 # 46 perdido: DESCONOCIDA retenida, no se entrega
        self.assertEqual(first, [])
        second = self.finish(5.5)                # 2 perdido: ENTRADA
        self.assertEqual([e.event_type for e in second], [GateEvent.ENTRADA])
        self.assertEqual(self.finish(9.0), [])   # el fragmento se descarta

    def test_desconocida_genuina_se_entrega_tras_la_espera(self):
        self.feed(9, [0.60], t0=0.0)             # visto una vez en el portal, sin evento cerca
        self.assertEqual(self.finish(2.5), [])
        ev = self.finish(6.0)
        self.assertEqual([e.event_type for e in ev], [GateEvent.DESCONOCIDA])


class SoloMovimientoTest(unittest.TestCase):
    """ROI_SOLO_MOVIMIENTO: cruces y desplazamientos claros salen normales; lo quieto sale MARCADO."""

    def setUp(self):
        self.eng = ROIEngine(line_a_y_ratio=LINE_A, line_b_y_ratio=LINE_B, track_lost_sec=2.0,
                             min_displacement_ratio=0.04, solo_movimiento=True)

    def feed(self, track_id, ys, t0=0.0, dt=0.3, x=0.4):
        for i, y in enumerate(ys):
            self.eng.update(track_id, bbox(y, x), H, W, timestamp=t0 + i * dt)

    def finish(self, at):
        return self.eng.collect_finished(at)

    def test_objeto_quieto_en_el_portal_sale_marcado_a_cualquier_posicion(self):
        # Un auto estacionado visto a través de la reja: y=0,60 sin moverse, en cualquier x
        for x in (0.08, 0.5, 0.9):
            self.feed(100 + int(x * 10), [0.60] * 12, x=x)
        self.finish(10)
        evs = self.finish(20)
        self.assertEqual(len(evs), 3)
        self.assertTrue(all(e.sin_movimiento and e.event_type == GateEvent.DESCONOCIDA for e in evs))
        self.assertEqual(self.eng.total_quietos, 3)

    def test_visto_una_vez_en_la_rampa_sale_marcado(self):
        self.feed(3, [0.92])
        evs = self.finish(10)
        self.assertEqual([(e.event_type, e.sin_movimiento) for e in evs], [(GateEvent.SALIDA, True)])

    def test_cruce_completo_sale_normal(self):
        self.feed(1, [0.45, 0.58, 0.64, 0.72, 0.80])
        evs = self.finish(10)
        self.assertEqual([(e.event_type, e.sin_movimiento) for e in evs], [(GateEvent.ENTRADA, False)])
        self.assertEqual(evs[0].reason, "CRUCE_COMPLETO")
        self.feed(2, [0.85, 0.72, 0.62, 0.50], t0=20.0)
        self.assertEqual([(e.event_type, e.sin_movimiento) for e in self.finish(30)], [(GateEvent.SALIDA, False)])

    def test_desplazamiento_claro_sale_normal(self):
        self.feed(4, [0.577, 0.64])
        evs = self.finish(10)
        self.assertEqual([(e.event_type, e.sin_movimiento) for e in evs], [(GateEvent.ENTRADA, False)])
        self.assertEqual(evs[0].reason, "TRACK_PERDIDO_DESPLAZAMIENTO")

    def test_objeto_quieto_no_contamina_la_continuacion_ni_los_fragmentos(self):
        # Un quieto no debe hacer que el auto siguiente se trate como "continuación" de un paso que no existió
        self.feed(7, [0.92])
        self.finish(10)
        self.assertIsNone(self.eng._last_event)
        self.assertEqual(len(self.eng._confirmed), 0)
        self.assertEqual(self.eng.total_exits, 0)

    def test_apagado_conserva_el_comportamiento_anterior(self):
        eng = ROIEngine(line_a_y_ratio=LINE_A, line_b_y_ratio=LINE_B, track_lost_sec=2.0, min_displacement_ratio=0.04)
        for i in range(12):
            eng.update(7, bbox(0.60, 0.1), H, W, timestamp=i * 0.3)
        eng.collect_finished(10)
        evs = eng.collect_finished(14)
        self.assertEqual([(e.event_type, e.sin_movimiento) for e in evs], [(GateEvent.DESCONOCIDA, False)])


if __name__ == "__main__":
    unittest.main()


class SoloCrucesTest(unittest.TestCase):
    """
    ROI_SOLO_CRUCES (regla estricta): solo avisa el cruce. Entrada = estuvo arriba de la línea B (calle) y llegó a
    la A. Salida = estuvo abajo de la A (interior) y llegó a la B. Todo lo demás sale MARCADO (se archiva, no
    va al panel). Un seguimiento cortado a mitad del portón se une si el track siguiente nace pegado y sigue el
    mismo sentido.
    """

    def setUp(self):
        self.eng = ROIEngine(line_a_y_ratio=LINE_A, line_b_y_ratio=LINE_B, track_lost_sec=2.0,
                             min_displacement_ratio=0.04, solo_cruces=True, union_sec=3.0)

    def feed(self, track_id, ys, t0=0.0, dt=0.3, x=0.4, xs=None):
        for i, y in enumerate(ys):
            self.eng.update(track_id, bbox(y, xs[i] if xs else x), H, W, timestamp=t0 + i * dt)

    def cerrar(self, t):
        # dos pasadas: la primera da por perdidos los tracks; la segunda archiva los que nadie continuó
        return self.eng.collect_finished(t) + self.eng.collect_finished(t + 10)

    def avisos(self, evs):
        return [(e.event_type, e.reason) for e in evs if not e.sin_movimiento]

    def archivados(self, evs):
        return [e for e in evs if e.sin_movimiento]

    def test_entrada_y_salida_completas_avisan(self):
        self.feed(1, [0.45, 0.52, 0.60, 0.66, 0.75])
        self.assertEqual(self.avisos(self.cerrar(10)), [(GateEvent.ENTRADA, "CRUCE_COMPLETO")])
        self.feed(2, [0.95, 0.85, 0.70, 0.60, 0.50], t0=30)
        self.assertEqual(self.avisos(self.cerrar(40)), [(GateEvent.SALIDA, "CRUCE_COMPLETO")])

    def test_suv_estacionado_detras_de_la_reja_nunca_avisa(self):
        # Posición real del SUV (19/11): borde inferior entre 0,58 y 0,62, a la izquierda; se pierde y reaparece
        # decenas de veces en el mismo lugar (con y sin unión de trayectorias).
        evs = []
        t = 0.0
        for k in range(40):
            self.feed(1000 + k, [0.606, 0.60, 0.612, 0.58, 0.61], t0=t, x=0.09)
            t += 1.5 + (k % 3) * 2.5           # a veces vuelve antes de 3 s (se une), a veces después
            evs += self.eng.collect_finished(t)
        evs += self.cerrar(t + 100)
        self.assertEqual(self.avisos(evs), [])
        self.assertTrue(len(self.archivados(evs)) >= 1)

    def test_auto_de_arriba_que_solo_toca_la_linea_b_no_avisa(self):
        self.feed(3, [0.40, 0.50, 0.57, 0.62, 0.66, 0.62, 0.55, 0.45])   # se asoma al portón y se va
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [])

    def test_auto_de_abajo_que_no_llega_a_la_linea_b_no_avisa(self):
        self.feed(4, [0.95, 0.85, 0.75, 0.66, 0.60])                 # sale pero se pierde antes de la B
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [])
        self.assertEqual(len(self.archivados(evs)), 1)               # queda archivado, no se pierde

    def test_entrada_con_seguimiento_cortado_se_une_y_avisa(self):
        # El tracker pierde al auto en la franja (reja/faros) y le da otro ID 0,8 s después, más abajo
        self.feed(5, [0.42, 0.50, 0.58, 0.63], dt=0.3)              # último punto t=0,9 en y=0,63
        self.eng.collect_finished(3.0)                               # se pierde (2 s sin verlo)
        self.feed(6, [0.66, 0.72, 0.80], t0=3.2, dt=0.3)             # nace pegado y sigue entrando
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [(GateEvent.ENTRADA, "CRUCE_COMPLETO")])
        self.assertEqual(self.eng.total_unidos, 1)
        self.assertEqual(self.archivados(evs), [])

    def test_salida_con_seguimiento_cortado_se_une_y_avisa(self):
        self.feed(7, [0.95, 0.85, 0.74, 0.66], dt=0.3)
        self.eng.collect_finished(3.0)
        self.feed(8, [0.62, 0.55, 0.48], t0=3.5, dt=0.3)
        self.assertEqual(self.avisos(self.cerrar(30)), [(GateEvent.SALIDA, "CRUCE_COMPLETO")])

    def test_no_se_une_si_esta_lejos_o_tarde(self):
        self.feed(9, [0.42, 0.50, 0.58, 0.63])
        self.eng.collect_finished(3.0)
        self.feed(10, [0.66, 0.72, 0.80], t0=3.2, x=0.85)           # nace en el otro extremo del cuadro
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [])
        self.feed(11, [0.42, 0.50, 0.58, 0.63], t0=100)
        self.eng.collect_finished(103.0)
        self.feed(12, [0.66, 0.72, 0.80], t0=107.5)                  # 6 s después: es otro vehículo
        self.assertEqual(self.avisos(self.cerrar(130)), [])

    def test_no_se_une_un_track_que_nace_en_la_calle_con_un_perdido_del_interior_que_ya_salio_a_la_calle(self):
        # El sentido importa: un perdido que estuvo en la calle no continúa en otro que nace en la calle
        self.feed(13, [0.45, 0.52, 0.60])
        self.eng.collect_finished(3.0)
        self.feed(14, [0.50, 0.45], t0=3.2)
        self.assertEqual(self.avisos(self.cerrar(30)), [])

    def test_lo_que_se_pierde_sin_cruzar_queda_archivado_con_su_ruta(self):
        self.feed(15, [0.577, 0.64])                                  # antes salía como ENTRADA por desplazamiento
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [])
        self.assertEqual([(e.event_type, e.reason) for e in self.archivados(evs)],
                         [(GateEvent.ENTRADA, "TRACK_PERDIDO_DESPLAZAMIENTO")])
        self.assertEqual(self.eng.total_sin_cruce, 1)

    def test_auto_que_entra_detras_de_otro_no_se_confunde(self):
        # Dos autos seguidos: el segundo entra completo; el primero ya avisó. Dos entradas, sin uniones raras.
        self.feed(16, [0.45, 0.58, 0.70, 0.82])
        self.feed(17, [0.42, 0.52, 0.60, 0.70, 0.80], t0=4.0)
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [(GateEvent.ENTRADA, "CRUCE_COMPLETO")] * 2)

    def test_dos_autos_que_se_cruzan_en_el_porton_no_inventan_avisos(self):
        # Uno sale y se pierde en la franja; justo nace otro en la franja que entra. La unión no puede fabricar
        # un cruce que ninguno de los dos hizo completo.
        self.feed(30, [0.95, 0.85, 0.74, 0.64])
        self.eng.collect_finished(3.0)
        self.feed(31, [0.62, 0.70, 0.80], t0=3.2)
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [])

    def test_unido_con_track_aun_activo_cuando_vence_el_viejo(self):
        # El tracker le da otro ID antes de que el viejo venza: el nuevo ya está activo cuando se evalúa el viejo
        self.feed(32, [0.42, 0.50, 0.58, 0.63], dt=0.3)              # último punto t=0,9
        self.feed(33, [0.66, 0.70], t0=1.2, dt=0.3)                  # nace a los 0,3 s, sigue visible
        self.eng.collect_finished(3.0)                               # el 32 vence (2,1 s sin verlo) y se une al 33
        self.feed(33, [0.78, 0.85], t0=3.1, dt=0.3)
        evs = self.cerrar(30)
        self.assertEqual(self.avisos(evs), [(GateEvent.ENTRADA, "CRUCE_COMPLETO")])
        self.assertEqual(self.eng.total_unidos, 1)
