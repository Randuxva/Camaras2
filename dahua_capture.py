"""Captura de frames desde cámara Dahua vía snapshot HTTP con sesión RPC2."""
import os
import time
from typing import Optional

import cv2
import numpy as np
import requests

from dahua_auth import dahua_base_url_from_env, dahua_credentials_from_env, dahua_login


class DahuaSnapshotCapture:
    """Lee snapshots periódicos (alternativa cuando OpenCV no abre MJPEG/RTSP)."""

    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        channel: int | None = None,
        interval_ms: int | None = None,
    ):
        self.base_url = (base_url or dahua_base_url_from_env() or "").rstrip("/")
        self.username, self.password = (
            (username, password)
            if username and password
            else dahua_credentials_from_env()
        )
        self.channel = channel or int(os.getenv("DAHUA_CHANNEL", "1"))
        self.interval_ms = interval_ms or int(os.getenv("DAHUA_SNAPSHOT_MS", "300"))
        self._session: requests.Session | None = None
        self._opened = False

    def open(self) -> bool:
        if not self.base_url or not self.password:
            return False
        try:
            self._session = dahua_login(self.base_url, self.username, self.password)
            test = self._fetch_snapshot_bytes()
            self._opened = test is not None
            return self._opened
        except Exception as exc:
            print(f"Error login Dahua: {exc}")
            self._opened = False
            return False

    def isOpened(self) -> bool:
        return self._opened

    def _fetch_snapshot_bytes(self) -> bytes | None:
        if not self._session:
            return None
        url = f"{self.base_url}/cgi-bin/snapshot.cgi?channel={self.channel}"
        response = self._session.get(url, timeout=10)
        if response.status_code != 200:
            return None
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and len(response.content) < 1000:
            return None
        return response.content

    def read(self):
        data = self._fetch_snapshot_bytes()
        if not data:
            return False, None
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        time.sleep(self.interval_ms / 1000.0)
        return True, frame

    def release(self):
        self._opened = False
        if self._session:
            self._session.close()
        self._session = None
