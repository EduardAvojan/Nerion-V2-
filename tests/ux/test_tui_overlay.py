def test_overlay_render_smoke():
    # This is a placeholder. Bind it to your overlay renderer once implemented.
    badges = {"eslint": 3, "tsc": 1, "other": 0}
    line = f"eslint:{badges['eslint']}  tsc:{badges['tsc']}  other:{badges['other']}"
    assert "eslint:" in line and "tsc:" in line
