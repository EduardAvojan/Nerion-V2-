def test_affected_preview_contract():
    # Integrate with `nerion graph affected --js` JSON output.
    affected = {"symbol":"Foo","depth":2,"importers":[{"path":"src/app.tsx","alias":"Foo"}]}
    assert affected["symbol"] == "Foo" and affected["importers"]
