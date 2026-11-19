"""Un evento no se descarta por cantidad de intentos, y los huecos de vigilancia se registran."""
import tempfile
import time
import unittest
from pathlib import Path

from cloud.offline_queue import OfflineQueue, retry_delay
from cloud.watch_gaps import WatchGapTracker


def _queue(tmp_path):
    return OfflineQueue(str(tmp_path / "q.db"))


def _enqueue(q, tmp_path, name="a.webp"):
    img = tmp_path / name
    img.write_bytes(b"x")
    return q.enqueue(track_id=1, event_type="ENTRADA", image_path=str(img), timestamp=int(time.time()),
                     camera_id="cam", sucursal_id="suc")


class _Reloj:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class NoPerderEventos(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_espera_creciente_con_tope(self):
        assert retry_delay(1) == 15
        assert retry_delay(2) == 30
        assert retry_delay(5) == 240
        assert retry_delay(50) == 1800

    def test_muchos_intentos_no_descartan_el_evento(self):
        """Caso real: la función respondió error durante horas y a los 10 intentos se perdían."""
        q = _queue(self.tmp)
        eid = _enqueue(q, self.tmp)
        for _ in range(60):
            q.increment_attempts(eid, error="HTTP 200: MODEL_ERROR sin cuota")
        assert q.count_pending() == 1
        assert q.count_failed() == 0
        # Sigue en cola pero esperando su turno: no se reintenta en cada vuelta
        assert q.get_pending() == []
        assert [e.id for e in q.get_pending(due_only=False)] == [eid]

    def test_solo_falla_por_antiguedad_y_se_puede_reencolar(self):
        q = _queue(self.tmp)
        eid = _enqueue(q, self.tmp)
        q.increment_attempts(eid, max_age_sec=-1)
        assert q.count_failed() == 1 and q.count_pending() == 0
        assert q.requeue_failed() == 1
        assert [e.id for e in q.get_pending()] == [eid]

    def test_reencolar_ignora_eventos_sin_foto(self):
        q = _queue(self.tmp)
        eid = _enqueue(q, self.tmp)
        q.increment_attempts(eid, max_age_sec=-1)
        (self.tmp / "a.webp").unlink()
        assert q.requeue_failed() == 0

    def test_los_eventos_salen_del_mas_viejo_al_mas_nuevo(self):
        q = _queue(self.tmp)
        a = _enqueue(q, self.tmp, "a.webp")
        time.sleep(0.01)
        b = _enqueue(q, self.tmp, "b.webp")
        assert [e.id for e in q.get_pending()] == [a, b]

    def test_suspension_de_la_pc_se_registra_como_hueco(self):
        q = _queue(self.tmp)
        reloj = _Reloj()
        g = WatchGapTracker(q, clock=reloj)
        assert g.tick(True) is None
        reloj.t += 15
        assert g.tick(True) is None
        reloj.t += 5 * 3600  # la PC se suspendió 5 horas
        gap = g.tick(True)
        assert gap and gap["reason"] == "SIN_REPORTE" and gap["to"] - gap["from"] == 5 * 3600
        assert len(q.unsent_gaps()) == 1

    def test_camara_sin_imagen_con_el_agente_vivo(self):
        q = _queue(self.tmp)
        reloj = _Reloj()
        g = WatchGapTracker(q, clock=reloj)
        g.tick(True)
        for _ in range(40):  # 10 minutos sin imagen, el agente sigue corriendo
            reloj.t += 15
            assert g.tick(False) is None
        reloj.t += 15
        gap = g.tick(True)
        assert gap and gap["reason"] == "SIN_VIDEO"

    def test_reinicio_del_agente_usa_la_ultima_marca_guardada(self):
        q = _queue(self.tmp)
        reloj = _Reloj()
        WatchGapTracker(q, clock=reloj).tick(True)
        reloj.t += 2 * 3600  # agente apagado dos horas
        gap = WatchGapTracker(q, clock=reloj).tick(True)
        assert gap and gap["reason"] == "SIN_REPORTE"

    def test_cortes_breves_no_son_huecos(self):
        q = _queue(self.tmp)
        reloj = _Reloj()
        g = WatchGapTracker(q, clock=reloj)
        g.tick(True)
        reloj.t += 60  # reconexión normal de la cámara
        assert g.tick(True) is None
        assert q.unsent_gaps() == []

    def test_huecos_enviados_no_se_reenvian(self):
        q = _queue(self.tmp)
        q.add_gap(100.0, 500.0, "SIN_REPORTE")
        ids = [x["id"] for x in q.unsent_gaps()]
        q.mark_gaps_sent(ids)
        assert q.unsent_gaps() == []

    def test_el_libro_conserva_cada_evento_y_lo_que_respondio_la_nube(self):
        q = _queue(self.tmp)
        a = _enqueue(q, self.tmp, "a.webp")
        b = _enqueue(q, self.tmp, "b.webp")
        q.increment_attempts(b, error="HTTP 200: MODEL_ERROR")
        q.mark_completed(a, {"success": True, "action": "CREATE_ENTRY", "data": {"movimiento": {"codigo": 41, "placaOCR": "AB123CD"}}})
        # La cola se vacía al subir; el libro no
        assert [e.id for e in q.get_pending(due_only=False)] == [b]
        libro = {e["id"]: e for e in q.event_log_rows()}
        assert libro[a]["status"] == "SUBIDO" and libro[a]["action"] == "CREATE_ENTRY"
        assert libro[a]["movimiento_codigo"] == 41 and libro[a]["plate"] == "AB123CD"
        assert libro[b]["status"] == "EN_COLA" and libro[b]["attempts"] == 1 and "MODEL_ERROR" in libro[b]["last_error"]

    def test_el_libro_marca_los_eventos_que_no_subieron(self):
        q = _queue(self.tmp)
        eid = _enqueue(q, self.tmp)
        q.increment_attempts(eid, max_age_sec=-1, error="sin respuesta")
        assert q.event_log_rows()[0]["status"] == "FALLIDO"
        q.requeue_failed()
        assert q.event_log_rows()[0]["status"] == "EN_COLA"


if __name__ == "__main__":
    unittest.main()
