"""
Prueba instrumentada del agente ANPR: corre main.main() sin cambiar su lógica y registra
cada inferencia, track, evento, webhook y métricas de sistema en AUDIT_DIR.
Uso (desde anpr-edge/): venv/Scripts/python.exe tools/audit_run.py <segundos>
En paralelo: venv/Scripts/python.exe tools/poll_db.py <segundos>, con el mismo AUDIT_DIR.
Los resultados (fotos y recortes con patentes = datos personales) quedan en audit_runs/, fuera de git.
"""
import json, os, sys, threading, time, queue as _queue, types, _thread
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT)); os.chdir(AGENT)
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 3600
AUDIT = Path(os.environ.get("AUDIT_DIR") or AGENT / "audit_runs" / time.strftime("%Y%m%d_%H%M")); (AUDIT / "frames").mkdir(parents=True, exist_ok=True); (AUDIT / "crops").mkdir(exist_ok=True)
T0 = time.time()

_locks = {}
def jl(name, obj):
    obj = {"t": round(time.time() - T0, 3), "wall": round(time.time(), 3), **obj}
    lk = _locks.setdefault(name, threading.Lock())
    with lk, open(AUDIT / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

import cv2, psutil
from config import cfg
import main as agent_main
from ultralytics import YOLO
from core.roi_engine import ROIEngine
from pipeline.consumer import InferenceConsumer
from cloud.uploader import CloudUploader
import cloud.uploader as upl_mod

H_LINE_A, H_LINE_B = cfg.roi_line_a_y_ratio, cfg.roi_line_b_y_ratio
stats = {"put": 0, "drop": 0, "get": 0, "infer": 0, "last_wall": None, "shape": None, "snap": 0}

# ── Cola: cuenta frames producidos/descartados y edad del frame al consumirse ─────────────
class AuditQueue(_queue.Queue):
    def put_nowait(self, item):
        stats["put"] += 1
        if stats["shape"] is None:
            stats["shape"] = item[0].shape; jl("info", {"frame_shape": item[0].shape})
        return super().put_nowait(item)
    def get_nowait(self):
        stats["drop"] += 1  # el productor saca el viejo para meter el nuevo
        return super().get_nowait()
    def get(self, *a, **k):
        item = super().get(*a, **k)
        stats["get"] += 1; stats["last_wall"] = item[1]
        return item
agent_main.queue = types.SimpleNamespace(Queue=AuditQueue, Empty=_queue.Empty)

# ── Snapshots anotados ───────────────────────────────────────────────────────────────
_last_snap = {"det": 0.0, "idle": 0.0, "motion": 0.0}
def snapshot(frame, boxes, ids, confs, kind):
    h, w = frame.shape[:2]; s = 960 / w
    img = cv2.resize(frame, (960, int(h * s)), interpolation=cv2.INTER_LINEAR)
    for y, c in ((H_LINE_B, (0, 255, 255)), (H_LINE_A, (0, 0, 255))):
        cv2.line(img, (0, int(y * h * s)), (960, int(y * h * s)), c, 1)
    for (x1, y1, x2, y2), tid, cf in zip(boxes, ids, confs):
        cv2.rectangle(img, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), (0, 255, 0), 2)
        cv2.putText(img, f"#{tid} {cf:.2f}", (int(x1 * s), max(12, int(y1 * s) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    ts = time.strftime("%H%M%S", time.localtime()) + f"{time.time() % 1:.1f}"[1:]
    cv2.putText(img, ts, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    name = f"{ts}_{kind}.jpg"
    cv2.imwrite(str(AUDIT / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 70]); stats["snap"] += 1
    return name

# ── Inferencia ───────────────────────────────────────────────────────────────────────
_orig_track = YOLO.track
def track(self, source, *a, **k):
    t0 = time.perf_counter(); age = time.time() - stats["last_wall"] if stats["last_wall"] else None
    res = _orig_track(self, source, *a, **k)
    ms = (time.perf_counter() - t0) * 1000; stats["infer"] += 1
    b = res[0].boxes; boxes, ids, confs, cls = [], [], [], []
    if b is not None and len(b):
        boxes = b.xyxy.int().cpu().tolist(); confs = [round(float(x), 3) for x in b.conf.cpu().tolist()]
        cls = b.cls.int().cpu().tolist(); ids = b.id.int().cpu().tolist() if b.id is not None else [None] * len(boxes)
    h = source.shape[0]
    dets = [{"id": i, "c": c, "conf": cf, "box": bx, "yb": round(bx[3] / h, 3)} for bx, i, cf, c in zip(boxes, ids, confs, cls)]
    snap = None; now = time.time()
    near = any(d["yb"] >= H_LINE_B - 0.06 for d in dets)
    if near and now - _last_snap["det"] >= 0.5:
        _last_snap["det"] = now; snap = snapshot(source, boxes, ids, confs, "det")
    elif not dets and now - _last_snap["motion"] >= 2.0:
        _last_snap["motion"] = now; snap = snapshot(source, [], [], [], "mov")
    jl("infer", {"ms": round(ms, 1), "age": round(age, 2) if age else None, "dets": dets, "snap": snap})
    return res
YOLO.track = track

_orig_motion = InferenceConsumer._check_motion
def check_motion(self, frame):
    r = _orig_motion(self, frame)
    now = time.time()
    if now - _last_snap["idle"] >= 60:
        _last_snap["idle"] = now; snapshot(frame, [], [], [], "idle")
    return r
InferenceConsumer._check_motion = check_motion

# ── ROI ──────────────────────────────────────────────────────────────────────────────
_orig_update = ROIEngine.update
def roi_update(self, track_id, bbox, frame_height, frame_width, timestamp):
    ev = _orig_update(self, track_id=track_id, bbox=bbox, frame_height=frame_height, frame_width=frame_width, timestamp=timestamp)
    if ev is not None:
        jl("roi", {"tid": track_id, "event": ev.event_type.value, "reason": ev.reason})
    return ev
ROIEngine.update = roi_update
_orig_collect = ROIEngine.collect_finished
def collect(self, now):
    evs = _orig_collect(self, now)
    for ev in evs:
        jl("roi", {"tid": ev.track_id, "event": ev.event_type.value, "reason": ev.reason, "deferred": True})
    return evs
ROIEngine.collect_finished = collect

# ── Despacho: guarda copia de los recortes (el agente los borra al confirmar) ─────────
_orig_save = CloudUploader.save_and_dispatch
def save_and_dispatch(self, crops, track_id, event_type, timestamp=None, reason="", **kw):
    names = []
    plates = kw.get("plate_crops") or []
    for kind, imgs in (("", crops), ("plate", plates)):
        for i, c in enumerate(imgs):
            n = f"{int(time.time())}_t{track_id}_{event_type.lower()}_{kind}{i}.jpg"
            cv2.imwrite(str(AUDIT / "crops" / n), c, [cv2.IMWRITE_JPEG_QUALITY, 92]); names.append(n)
    jl("dispatch", {"tid": track_id, "event": event_type, "reason": reason, "n": len(crops),
                    "n_plates": len(plates), "local_ocr": kw.get("local_ocr"),
                    "sizes": [list(c.shape[:2]) for c in crops], "crops": names})
    return _orig_save(self, crops, track_id, event_type, timestamp, reason, **kw)
CloudUploader.save_and_dispatch = save_and_dispatch

# ── Webhook: payload y respuesta completa (sin headers) ──────────────────────────────
_real_requests = upl_mod.requests
class ReqShim:
    def __getattr__(self, n): return getattr(_real_requests, n)
    def post(self, url, json=None, headers=None, timeout=None, **k):
        t0 = time.perf_counter()
        try:
            r = _real_requests.post(url, json=json, headers=headers, timeout=timeout, **k)
            body = r.text
            jl("webhook", {"type": (json or {}).get("type", "event"), "event_id": (json or {}).get("event_id"),
                           "dir": (json or {}).get("direction"), "tid": (json or {}).get("track_id"),
                           "status": r.status_code, "ms": round((time.perf_counter() - t0) * 1000), "body": body[:4000]})
            return r
        except Exception as e:
            jl("webhook", {"type": (json or {}).get("type", "event"), "event_id": (json or {}).get("event_id"),
                           "error": repr(e), "ms": round((time.perf_counter() - t0) * 1000)})
            raise
upl_mod.requests = ReqShim()

# ── Métricas de sistema cada 5 s ─────────────────────────────────────────────────────
def metrics():
    p = psutil.Process(); p.cpu_percent(); psutil.cpu_percent()
    last = dict(stats); lt = time.time()
    while True:
        time.sleep(5); now = time.time(); dt = now - lt
        jl("metrics", {"cpu_proc": p.cpu_percent(), "cpu_sys": psutil.cpu_percent(), "rss_mb": round(p.memory_info().rss / 1e6),
                       "fps_in": round((stats["put"] - last["put"]) / dt, 2), "fps_infer": round((stats["infer"] - last["infer"]) / dt, 2),
                       "put": stats["put"], "infer": stats["infer"], "snaps": stats["snap"],
                       "net_recv_kb": round(psutil.net_io_counters().bytes_recv / 1e3)})
        last = dict(stats); lt = now
threading.Thread(target=metrics, daemon=True).start()

def stopper():
    time.sleep(DURATION); jl("info", {"msg": "fin de la prueba"}); _thread.interrupt_main()
threading.Thread(target=stopper, daemon=True).start()

jl("info", {"msg": "inicio", "duration": DURATION, "model": cfg.model_name, "imgsz": cfg.yolo_imgsz,
            "line_a": H_LINE_A, "line_b": H_LINE_B, "conf": cfg.yolo_confidence, "secret": bool(cfg.agent_secret)})
try:
    agent_main.main()
except KeyboardInterrupt:
    pass
jl("info", {"msg": "agente detenido", "stats": {k: v for k, v in stats.items() if k != "shape"}})
