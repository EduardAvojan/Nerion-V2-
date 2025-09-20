from core.http.schemas import ok, err

def test_ok():
    r = ok({"x":1})
    assert r["ok"] is True and r["data"]["x"] == 1 and r["errors"] == []

def test_err():
    r = err("BAD", "bad input", "endpoint")
    assert r["ok"] is False and r["errors"][0]["code"] == "BAD"
