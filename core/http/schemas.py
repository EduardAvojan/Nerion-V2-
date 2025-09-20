from typing import Any, Dict

def ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "data": data, "errors": []}

def err(code: str, msg: str, where: str = "") -> Dict[str, Any]:
    return {"ok": False, "data": {}, "errors": [{"code": code, "msg": msg, "where": where}]}
