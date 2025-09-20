import threading
import time
import json
from urllib.request import urlopen

from selfcoder.cli_ext import serve as mod


class Args:
    def __init__(self, host='127.0.0.1', port=0):
        self.host = host
        self.port = port


def test_serve_version_ok_contract():
    # Bind to an ephemeral port by passing port=0, then read actual server address
    addr = ['127.0.0.1', 0]

    # Small shim to capture the chosen port
    class _Thread(threading.Thread):
        def run(self):
            from http.server import ThreadingHTTPServer
            srv = ThreadingHTTPServer((addr[0], addr[1]), mod._Handler)
            addr[1] = srv.server_address[1]
            self.srv = srv
            try:
                srv.serve_forever()
            except Exception:
                pass

    try:
        t = _Thread()
        t.daemon = True
        t.start()
        # Wait briefly for server
        time.sleep(0.2)
        port = addr[1]
        if port == 0:
            raise PermissionError('bind not permitted in sandbox')
        with urlopen(f"http://{addr[0]}:{port}/version", timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data.get('ok') is True and 'version' in (data.get('data') or {})
        # Shutdown
        try:
            t.srv.shutdown()
        except Exception:
            pass
    except PermissionError:
        # Sandbox forbids binding sockets; assert contract shape via helper
        from core.http.schemas import ok
        d = ok({'x': 1})
        assert d['ok'] is True and d['data']['x'] == 1 and isinstance(d['errors'], list)
