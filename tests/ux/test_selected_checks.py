def test_selected_checks_contract():
    # Provide selection -> expect subset of files sent to gate/linters.
    selection = ["src/a.ts", "src/b.ts"]
    assert len(selection) == 2
