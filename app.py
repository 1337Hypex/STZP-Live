
import argparse
import json
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None


COCO_VEHICLES = {"car", "truck", "bus", "motorcycle"}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, col_name: str, decl: str) -> None:
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col_name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {decl}")
        conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT,
            source_mode TEXT NOT NULL DEFAULT 'direct',
            source TEXT,
            page_url TEXT,
            camera_rect_json TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            ref_w INTEGER NOT NULL DEFAULT 1280,
            ref_h INTEGER NOT NULL DEFAULT 720,
            map_center_lat REAL,
            map_center_lng REAL,
            map_zoom INTEGER,
            det_conf REAL DEFAULT 0.35,
            overlap REAL DEFAULT 0.22,
            infer_fps REAL DEFAULT 1.2,
            display_fps REAL DEFAULT 10.0,
            mjpeg_q INTEGER DEFAULT 80,
            browser_width INTEGER DEFAULT 1280,
            browser_height INTEGER DEFAULT 720,
            capture_interval REAL DEFAULT 1.0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spots (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            poly_json TEXT NOT NULL,
            lat REAL,
            lng REAL,
            FOREIGN KEY(camera_id) REFERENCES cameras(id)
        )
        """
    )
    conn.commit()

    # compatibility if user already has an older DB
    ensure_column(conn, "cameras", "source_mode", "source_mode TEXT NOT NULL DEFAULT 'direct'")
    ensure_column(conn, "cameras", "source", "source TEXT")
    ensure_column(conn, "cameras", "page_url", "page_url TEXT")
    ensure_column(conn, "cameras", "camera_rect_json", "camera_rect_json TEXT")
    ensure_column(conn, "cameras", "browser_width", "browser_width INTEGER DEFAULT 1280")
    ensure_column(conn, "cameras", "browser_height", "browser_height INTEGER DEFAULT 720")
    ensure_column(conn, "cameras", "capture_interval", "capture_interval REAL DEFAULT 1.0")
    ensure_column(conn, "cameras", "det_conf", "det_conf REAL DEFAULT 0.35")
    ensure_column(conn, "cameras", "overlap", "overlap REAL DEFAULT 0.22")
    ensure_column(conn, "cameras", "infer_fps", "infer_fps REAL DEFAULT 1.2")
    ensure_column(conn, "cameras", "display_fps", "display_fps REAL DEFAULT 10.0")
    ensure_column(conn, "cameras", "mjpeg_q", "mjpeg_q INTEGER DEFAULT 80")


@dataclass
class CameraCfg:
    id: str
    name: str
    source_mode: str
    source: Optional[str]
    page_url: Optional[str]
    camera_rect: Optional[dict]
    enabled: bool
    ref_w: int
    ref_h: int
    map_center: Optional[list]
    map_zoom: int
    det_conf: float
    overlap: float
    infer_fps: float
    display_fps: float
    mjpeg_q: int
    browser_width: int
    browser_height: int
    capture_interval: float


def row_to_cfg(row) -> CameraCfg:
    rect = json.loads(row["camera_rect_json"]) if row["camera_rect_json"] else None
    center = None
    if row["map_center_lat"] is not None and row["map_center_lng"] is not None:
        center = [float(row["map_center_lat"]), float(row["map_center_lng"])]
    return CameraCfg(
        id=row["id"],
        name=row["name"] or row["id"],
        source_mode=(row["source_mode"] or "direct").lower(),
        source=row["source"],
        page_url=row["page_url"],
        camera_rect=rect,
        enabled=bool(row["enabled"]),
        ref_w=int(row["ref_w"]),
        ref_h=int(row["ref_h"]),
        map_center=center,
        map_zoom=int(row["map_zoom"]) if row["map_zoom"] is not None else 18,
        det_conf=float(row["det_conf"] if row["det_conf"] is not None else 0.35),
        overlap=float(row["overlap"] if row["overlap"] is not None else 0.22),
        infer_fps=float(row["infer_fps"] if row["infer_fps"] is not None else 1.2),
        display_fps=float(row["display_fps"] if row["display_fps"] is not None else 10.0),
        mjpeg_q=int(row["mjpeg_q"] if row["mjpeg_q"] is not None else 80),
        browser_width=int(row["browser_width"] if row["browser_width"] is not None else 1280),
        browser_height=int(row["browser_height"] if row["browser_height"] is not None else 720),
        capture_interval=float(row["capture_interval"] if row["capture_interval"] is not None else 1.0),
    )


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode screenshot bytes")
    return frame


def crop_with_rect(frame: np.ndarray, rect: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    x = max(0, min(w - 1, int(rect.get("x", 0))))
    y = max(0, min(h - 1, int(rect.get("y", 0))))
    rw = max(1, min(w - x, int(rect.get("w", 1))))
    rh = max(1, min(h - y, int(rect.get("h", 1))))
    return frame[y:y + rh, x:x + rw].copy()


def scale_polygon(poly: List[List[int]], src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    pts = np.array([[int(round(x * sx)), int(round(y * sy))] for x, y in poly], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


class WebPageGrabber:
    def __init__(self, page_url: str, width: int, height: int):
        self.page_url = page_url
        self.width = width
        self.height = height
        self._pw = None
        self.browser = None
        self.page = None

    def open(self):
        if sync_playwright is None:
            raise RuntimeError("Playwright is not installed. Run: python -m playwright install chromium")
        if self.page is not None:
            return
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True)
        page = self.browser.new_page(viewport={"width": self.width, "height": self.height})
        page.goto(self.page_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2.0)
        self.page = page

    def screenshot(self) -> bytes:
        self.open()
        return self.page.screenshot(type="jpeg", quality=85, full_page=False)

    def close(self):
        try:
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        try:
            if self.browser is not None:
                self.browser.close()
        except Exception:
            pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self.page = None
        self.browser = None
        self._pw = None


class Spot:
    def __init__(self, sid: str, poly_pts: np.ndarray, frame_w: int, frame_h: int, lat: Optional[float], lng: Optional[float]):
        self.id = sid
        self.poly_pts = poly_pts
        self.lat = lat
        self.lng = lng
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_pts], 255)
        self.mask = mask
        self.area = int(np.count_nonzero(mask))
        self.history = deque(maxlen=5)
        self.occupied = False
        self.conf = 0.0


class ParkingEngine:
    def __init__(self, conn: sqlite3.Connection, camera_id: str, model: YOLO):
        self.conn = conn
        self.camera_id = camera_id
        self.model = model

        self._lock = threading.Lock()
        self._running = False
        self._reload = True
        self._thread = None

        self.cfg: Optional[CameraCfg] = None
        self.spots: Dict[str, Spot] = {}
        self._spot_rows = []
        self.cap = None
        self.webpage = None
        self.frame_w = None
        self.frame_h = None
        self.last_jpeg = None
        self.last_error = None
        self.last_frame_ts = 0.0
        self.state = {"camera_id": camera_id, "name": camera_id, "ts": utc_iso(), "map": None, "spots": [], "error": None}

    def request_reload(self):
        self._reload = True

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_sources()

    def _close_sources(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        if self.webpage is not None:
            self.webpage.close()
        self.webpage = None

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "name": self.cfg.name if self.cfg else self.camera_id,
                "enabled": bool(self.cfg.enabled) if self.cfg else False,
                "source_mode": self.cfg.source_mode if self.cfg else "direct",
                "last_error": self.last_error,
                "last_frame_ts": self.last_frame_ts,
                "spots_count": len(self.spots),
            }

    def get_state(self) -> dict:
        with self._lock:
            return dict(self.state)

    def get_last_jpeg(self):
        with self._lock:
            return self.last_jpeg

    def _load_cfg(self):
        row = self.conn.execute("SELECT * FROM cameras WHERE id=?", (self.camera_id,)).fetchone()
        if row is None:
            self.cfg = None
            self.last_error = "Camera not found"
            self._spot_rows = []
            return
        self.cfg = row_to_cfg(row)
        self._spot_rows = self.conn.execute("SELECT * FROM spots WHERE camera_id=? ORDER BY id", (self.camera_id,)).fetchall()
        self.last_error = None

    def _open_source(self):
        self._close_sources()
        self.frame_w = None
        self.frame_h = None
        self.spots = {}
        if self.cfg is None or not self.cfg.enabled:
            return

        if self.cfg.source_mode == "webpage":
            if not self.cfg.page_url:
                self.last_error = "No page_url"
                return
            if not self.cfg.camera_rect:
                self.last_error = "No camera rect selected"
                return
            try:
                self.webpage = WebPageGrabber(self.cfg.page_url, self.cfg.browser_width, self.cfg.browser_height)
                self.webpage.open()
            except Exception as e:
                self.last_error = f"Webpage mode failed: {e}"
                self.webpage = None
                return
        else:
            if not self.cfg.source:
                self.last_error = "No direct source"
                return
            cap = cv2.VideoCapture(self.cfg.source)
            if not cap.isOpened():
                self.last_error = "Cannot open source"
                return
            self.cap = cap

        self.last_error = None

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.cfg is None:
            return None

        if self.cfg.source_mode == "webpage":
            if self.webpage is None:
                return None
            img_bytes = self.webpage.screenshot()
            full = decode_image_bytes(img_bytes)
            return crop_with_rect(full, self.cfg.camera_rect)

        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
            ok, frame = self.cap.read()
        return frame if ok else None

    def _init_spots(self, frame_w: int, frame_h: int):
        self.spots = {}
        if self.cfg is None:
            return
        for r in self._spot_rows:
            sid = r["id"]
            poly = json.loads(r["poly_json"])
            pts = scale_polygon(poly, self.cfg.ref_w, self.cfg.ref_h, frame_w, frame_h)
            lat = float(r["lat"]) if r["lat"] is not None else None
            lng = float(r["lng"]) if r["lng"] is not None else None
            self.spots[sid] = Spot(sid, pts, frame_w, frame_h, lat, lng)

    def _update_state(self):
        cfg = self.cfg
        self.state = {
            "camera_id": self.camera_id,
            "name": cfg.name if cfg else self.camera_id,
            "ts": utc_iso(),
            "map": {"center": cfg.map_center, "zoom": cfg.map_zoom} if cfg and cfg.map_center else None,
            "spots": [
                {"id": s.id, "occupied": bool(s.occupied), "conf": float(round(s.conf, 3)), "lat": s.lat, "lng": s.lng}
                for s in self.spots.values()
            ],
            "error": self.last_error,
        }

    def _loop(self):
        last_infer = 0.0
        last_disp = 0.0

        while self._running:
            if self._reload:
                self._reload = False
                self._load_cfg()
                self._open_source()
                with self._lock:
                    self._update_state()

            cfg = self.cfg
            if cfg is None or not cfg.enabled:
                time.sleep(0.2)
                continue

            try:
                frame = self._read_frame()
            except Exception as e:
                self.last_error = str(e)
                frame = None

            if frame is None:
                time.sleep(0.25)
                continue

            now = time.time()
            self.last_frame_ts = now

            if self.frame_w is None:
                self.frame_h, self.frame_w = frame.shape[:2]
                self._init_spots(self.frame_w, self.frame_h)

            if now - last_disp >= (1.0 / max(1.0, cfg.display_fps)):
                last_disp = now
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.mjpeg_q)])
                if ok2:
                    with self._lock:
                        self.last_jpeg = buf.tobytes()

            if now - last_infer < (1.0 / max(0.1, cfg.infer_fps)):
                time.sleep(max(0.01, cfg.capture_interval if cfg.source_mode == "webpage" else 0.0))
                continue
            last_infer = now

            try:
                results = self.model.predict(frame, conf=cfg.det_conf, verbose=False)[0]
                names = results.names
                boxes = []
                if results.boxes is not None and len(results.boxes) > 0:
                    xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
                    confs = results.boxes.conf.cpu().numpy()
                    clss = results.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                        label = names.get(int(k), str(int(k)))
                        if label in COCO_VEHICLES:
                            x1 = max(0, min(self.frame_w - 1, x1))
                            x2 = max(0, min(self.frame_w, x2))
                            y1 = max(0, min(self.frame_h - 1, y1))
                            y2 = max(0, min(self.frame_h, y2))
                            if x2 > x1 and y2 > y1:
                                boxes.append((x1, y1, x2, y2, float(c)))

                for s in self.spots.values():
                    occ_conf = 0.0
                    occ = False
                    if s.area <= 0:
                        s.history.append(0)
                        continue

                    for x1, y1, x2, y2, c in boxes:
                        inter = int(np.count_nonzero(s.mask[y1:y2, x1:x2]))
                        ratio = inter / float(s.area)
                        if ratio >= cfg.overlap:
                            occ = True
                            occ_conf = max(occ_conf, min(1.0, (ratio - cfg.overlap) / (1.0 - cfg.overlap)) * c)

                    s.history.append(1 if occ else 0)
                    smoothed = sum(s.history) >= (len(s.history) // 2 + 1)
                    s.occupied = smoothed
                    s.conf = occ_conf if smoothed else (1.0 - occ_conf)
            except Exception as e:
                self.last_error = f"Inference error: {e}"

            with self._lock:
                self._update_state()

            if cfg.source_mode == "webpage":
                time.sleep(max(0.05, cfg.capture_interval))

        self._close_sources()


class EngineManager:
    def __init__(self, conn: sqlite3.Connection, model: YOLO):
        self.conn = conn
        self.model = model
        self._lock = threading.Lock()
        self.engines: Dict[str, ParkingEngine] = {}

    def ensure(self, cid: str) -> ParkingEngine:
        with self._lock:
            if cid in self.engines:
                return self.engines[cid]
            e = ParkingEngine(self.conn, cid, self.model)
            e.start()
            self.engines[cid] = e
            return e

    def start_enabled(self):
        for r in self.conn.execute("SELECT id FROM cameras WHERE enabled=1 ORDER BY id").fetchall():
            self.ensure(r["id"])

    def stop(self, cid: str):
        with self._lock:
            e = self.engines.get(cid)
            if e:
                e.stop()
                del self.engines[cid]

    def reload(self, cid: str):
        self.ensure(cid).request_reload()

    def states(self):
        with self._lock:
            return [e.get_state() for e in self.engines.values()]

    def stats(self):
        with self._lock:
            return [e.get_stats() for e in self.engines.values()]


def mjpeg_from_engine(engine: ParkingEngine, target_fps: float = 10.0):
    delay = 1.0 / max(1.0, float(target_fps))
    while True:
        jpg = engine.get_last_jpeg()
        if jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                jpg + b"\r\n"
            )
        time.sleep(delay)


def capture_full_page_snapshot(page_url: str, width: int, height: int) -> bytes:
    grabber = WebPageGrabber(page_url, width, height)
    try:
        raw = grabber.screenshot()
        frame = decode_image_bytes(raw)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("Could not encode page snapshot")
        return buf.tobytes()
    finally:
        grabber.close()


async def asyncio_sleep(sec: float):
    import asyncio
    await asyncio.sleep(sec)


def create_app(db_path: str) -> FastAPI:
    conn = db_connect(db_path)
    init_db(conn)

    model = YOLO("yolov8n.pt")
    mgr = EngineManager(conn, model)
    mgr.start_enabled()

    app = FastAPI(title="Smart Parking – direct + webpage mode")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"ok": True, "ts": utc_iso()}

    @app.get("/api/cameras")
    def list_cameras():
        return [dict(r) for r in conn.execute("SELECT * FROM cameras ORDER BY id").fetchall()]

    @app.get("/api/cameras/{cid}")
    def get_camera(cid: str):
        r = conn.execute("SELECT * FROM cameras WHERE id=?", (cid,)).fetchone()
        if r is None:
            raise HTTPException(404, "camera not found")
        d = dict(r)
        d["camera_rect"] = json.loads(d["camera_rect_json"]) if d.get("camera_rect_json") else None
        return d

    @app.post("/api/cameras")
    def upsert_camera(payload: dict):
        mode = str(payload.get("source_mode", "direct")).strip().lower()
        if mode not in ("direct", "webpage"):
            raise HTTPException(400, "source_mode must be direct or webpage")

        cid = str(payload.get("id", "")).strip()
        if not cid:
            raise HTTPException(400, "missing id")

        source = str(payload.get("source", "")).strip() or None
        page_url = str(payload.get("page_url", "")).strip() or None

        if mode == "direct" and not source:
            raise HTTPException(400, "direct mode needs source")
        if mode == "webpage" and not page_url:
            raise HTTPException(400, "webpage mode needs page_url")

        rect = payload.get("camera_rect")
        rect_json = json.dumps(rect, ensure_ascii=False) if rect else None

        enabled = 1 if bool(payload.get("enabled", True)) else 0
        ref_w = int(payload.get("ref_w", 1280))
        ref_h = int(payload.get("ref_h", 720))
        det_conf = float(payload.get("det_conf", 0.35))
        overlap = float(payload.get("overlap", 0.22))
        infer_fps = float(payload.get("infer_fps", 1.2))
        display_fps = float(payload.get("display_fps", 10.0))
        mjpeg_q = int(payload.get("mjpeg_q", 80))
        browser_width = int(payload.get("browser_width", 1280))
        browser_height = int(payload.get("browser_height", 720))
        capture_interval = float(payload.get("capture_interval", 1.0))

        mc = payload.get("map_center")
        mz = payload.get("map_zoom")
        mlat = mlng = None
        if isinstance(mc, list) and len(mc) == 2:
            mlat, mlng = float(mc[0]), float(mc[1])
        mz = int(mz) if mz is not None else None

        conn.execute(
            "INSERT INTO cameras(id,name,source_mode,source,page_url,camera_rect_json,enabled,ref_w,ref_h,map_center_lat,map_center_lng,map_zoom,det_conf,overlap,infer_fps,display_fps,mjpeg_q,browser_width,browser_height,capture_interval) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, source_mode=excluded.source_mode, source=excluded.source, page_url=excluded.page_url, "
            "camera_rect_json=excluded.camera_rect_json, enabled=excluded.enabled, ref_w=excluded.ref_w, ref_h=excluded.ref_h, "
            "map_center_lat=excluded.map_center_lat, map_center_lng=excluded.map_center_lng, map_zoom=excluded.map_zoom, det_conf=excluded.det_conf, "
            "overlap=excluded.overlap, infer_fps=excluded.infer_fps, display_fps=excluded.display_fps, mjpeg_q=excluded.mjpeg_q, "
            "browser_width=excluded.browser_width, browser_height=excluded.browser_height, capture_interval=excluded.capture_interval",
            (
                cid, payload.get("name"), mode, source, page_url, rect_json, enabled, ref_w, ref_h, mlat, mlng, mz,
                det_conf, overlap, infer_fps, display_fps, mjpeg_q, browser_width, browser_height, capture_interval
            ),
        )
        conn.commit()

        if enabled:
            mgr.ensure(cid)
            mgr.reload(cid)
        else:
            mgr.stop(cid)

        return {"ok": True}

    @app.post("/api/cameras/{cid}/camera_rect")
    def save_camera_rect(cid: str, payload: dict):
        rect = payload.get("camera_rect")
        if not isinstance(rect, dict):
            raise HTTPException(400, "camera_rect must be object")
        conn.execute("UPDATE cameras SET camera_rect_json=? WHERE id=?", (json.dumps(rect, ensure_ascii=False), cid))
        conn.commit()
        mgr.reload(cid)
        return {"ok": True}

    @app.get("/api/cameras/{cid}/page_snapshot")
    def page_snapshot(cid: str):
        row = conn.execute("SELECT * FROM cameras WHERE id=?", (cid,)).fetchone()
        if row is None:
            raise HTTPException(404, "camera not found")
        cfg = row_to_cfg(row)
        if not cfg.page_url:
            raise HTTPException(400, "camera has no page_url")
        try:
            jpg = capture_full_page_snapshot(cfg.page_url, cfg.browser_width, cfg.browser_height)
        except Exception as e:
            raise HTTPException(500, str(e))
        return Response(content=jpg, media_type="image/jpeg")

    @app.delete("/api/cameras/{cid}")
    def delete_camera(cid: str):
        conn.execute("DELETE FROM spots WHERE camera_id=?", (cid,))
        conn.execute("DELETE FROM cameras WHERE id=?", (cid,))
        conn.commit()
        mgr.stop(cid)
        return {"ok": True}

    @app.get("/api/cameras/{cid}/mjpeg")
    def camera_mjpeg(cid: str, fps: float = Query(10.0)):
        engine = mgr.ensure(cid)
        engine.request_reload()
        return StreamingResponse(mjpeg_from_engine(engine, target_fps=fps), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/spots")
    def list_spots(camera_id_q: Optional[str] = None):
        if camera_id_q:
            rows = conn.execute("SELECT * FROM spots WHERE camera_id=? ORDER BY id", (camera_id_q,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM spots ORDER BY camera_id, id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["poly"] = json.loads(d.pop("poly_json"))
            out.append(d)
        return out

    @app.post("/api/cameras/{cid}/spots")
    def replace_spots(cid: str, payload: dict):
        spots = payload.get("spots")
        if not isinstance(spots, list):
            raise HTTPException(400, "payload.spots must be list")
        conn.execute("DELETE FROM spots WHERE camera_id=?", (cid,))
        for s in spots:
            sid = str(s.get("id", "")).strip()
            poly = s.get("poly")
            if not sid or not isinstance(poly, list) or len(poly) < 3:
                raise HTTPException(400, "spot needs id+poly>=3")
            lat = s.get("lat")
            lng = s.get("lng")
            conn.execute(
                "INSERT OR REPLACE INTO spots(id,camera_id,poly_json,lat,lng) VALUES(?,?,?,?,?)",
                (sid, cid, json.dumps(poly, ensure_ascii=False), float(lat) if lat is not None else None, float(lng) if lng is not None else None),
            )
        conn.commit()
        mgr.reload(cid)
        return {"ok": True, "count": len(spots)}

    @app.patch("/api/spots/{sid}")
    def update_spot(sid: str, payload: dict):
        lat = payload.get("lat")
        lng = payload.get("lng")
        if lat is None or lng is None:
            raise HTTPException(400, "need lat,lng")
        row = conn.execute("SELECT camera_id FROM spots WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise HTTPException(404, "spot not found")
        conn.execute("UPDATE spots SET lat=?, lng=? WHERE id=?", (float(lat), float(lng), sid))
        conn.commit()
        mgr.reload(row["camera_id"])
        return {"ok": True}

    @app.get("/api/stats")
    def stats():
        return {"ts": utc_iso(), "engines": mgr.stats()}

    @app.get("/api/state")
    def state():
        return {"ts": utc_iso(), "cameras": mgr.states()}

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json({"ts": utc_iso(), "cameras": mgr.states()})
                await asyncio_sleep(0.5)
        except WebSocketDisconnect:
            return

    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="parking.db")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.db)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
