"""Autenticación Dahua vía RPC2_Login para streams CGI."""
import hashlib
import os
from urllib.parse import urlparse

import requests


def _md5_hex(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def dahua_login(
    base_url: str,
    username: str,
    password: str,
    timeout: int = 10,
) -> requests.Session:
    """
    Login Dahua challenge-response.
    Retorna Session con cookies para usar en snapshot.cgi / mjpg.
    """
    session = requests.Session()
    login_url = base_url.rstrip("/") + "/RPC2_Login"

    first = session.post(
        login_url,
        json={
            "method": "global.login",
            "params": {
                "userName": username,
                "password": "",
                "clientType": "Web3.0",
                "loginType": "Direct",
                "authorityType": "Default",
                "passwordType": "Default",
            },
            "id": 1,
            "session": 0,
        },
        timeout=timeout,
    )
    first.raise_for_status()
    data = first.json()
    params = data.get("params") or {}
    realm = params.get("realm", "")
    random_nonce = params.get("random", "")
    rpc_session = data.get("session", 0)

    if not realm or not random_nonce:
        raise RuntimeError(f"Login Dahua paso 1 falló: {data}")

    inner = _md5_hex(f"{username}:{realm}:{password}")
    password_hash = _md5_hex(f"{username}:{random_nonce}:{inner}")

    second = session.post(
        login_url,
        json={
            "method": "global.login",
            "params": {
                "userName": username,
                "password": password_hash,
                "clientType": "Web3.0",
                "loginType": "Direct",
                "authorityType": "Default",
                "passwordType": "Default",
            },
            "id": 2,
            "session": rpc_session,
        },
        timeout=timeout,
    )
    second.raise_for_status()
    result = second.json()
    if not result.get("result"):
        params = result.get("params") or {}
        lock_sec = params.get("remainLockSecond")
        if lock_sec:
            raise RuntimeError(
                f"Cuenta bloqueada temporalmente ({lock_sec}s). "
                "Verifica usuario/clave o espera antes de reintentar."
            )
        raise RuntimeError(f"Login Dahua paso 2 falló: {result}")

    session.cookies.set("DhWebClientSessionID", str(rpc_session))
    session.headers.update({"X-Requested-With": "XMLHttpRequest"})
    return session


def dahua_base_url_from_env() -> str | None:
    host = os.getenv("DAHUA_HOST", "").strip()
    if host:
        port = os.getenv("DAHUA_PORT", "80").strip()
        scheme = os.getenv("DAHUA_SCHEME", "http").strip()
        return f"{scheme}://{host}:{port}"

    source = os.getenv("CAMERA_SOURCE", "").strip()
    if source.startswith("http"):
        parsed = urlparse(source)
        if parsed.hostname:
            port = parsed.port or 80
            scheme = parsed.scheme or "http"
            return f"{scheme}://{parsed.hostname}:{port}"
    return None


def dahua_credentials_from_env() -> tuple[str, str]:
    user = os.getenv("DAHUA_USER", os.getenv("CAMERA_USER", "admin"))
    password = os.getenv("DAHUA_PASS", os.getenv("CAMERA_PASS", ""))
    return user, password
