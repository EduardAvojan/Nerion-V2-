import os
import json
from pathlib import Path


def test_plugins_load_and_register(tmp_path, monkeypatch):
    # 1) Create a temporary plugin directory with a simple Plugin
    plugins_dir = tmp_path / "plugins"
    demo_dir = plugins_dir / "demo"
    demo_dir.mkdir(parents=True)

    plugin_py = demo_dir / "plugin.py"
    marker_path = tmp_path / "transformer_called.txt"

    plugin_py.write_text(
        f"""
class Plugin:
    name = "demo"
    version = "0.0.1"

    def register_transformers(self, registry):
        def _xf(tree, action):
            # record invocation by writing a marker file path passed via action
            mp = action.get('marker')
            if mp:
                try:
                    with open(mp, 'w', encoding='utf-8') as f:
                        f.write('called')
                except Exception:
                    pass
            return tree
        registry.add_transformer('my_action', _xf)

    def register_cli(self, add_cli_extension):
        def _extend(subparsers):
            sp = subparsers.add_parser('hello', help='demo plugin command')
            def _run(args):
                # set a flag on args so test can assert
                setattr(args, '_ran', True)
            sp.set_defaults(func=_run)
        add_cli_extension(_extend)
""",
        encoding="utf-8",
    )

    # 2) Point Nerion to this plugins directory
    monkeypatch.setenv("NERION_PLUGINS_DIR", str(plugins_dir))

    # 3) Load plugins programmatically
    from plugins.loader import load_plugins_auto
    load_plugins_auto()  # uses NERION_PLUGINS_DIR

    # 4) Verify CLI extension is wired: build the parser and run the command
    import selfcoder.cli as cli
    parser = cli._build_parser()
    ns = parser.parse_args(["plugins", "watch", "--plugins-dir", str(plugins_dir)])  # ensure base works

    # The plugin-added command should also be present
    ns = parser.parse_args(["hello"])  # parse plugin subcommand
    assert hasattr(ns, "func")
    ns.func(ns)
    assert getattr(ns, "_ran", False) is True

    # 5) Verify transformer is invoked via orchestrator action plumbing
    target_py = tmp_path / "mod.py"
    target_py.write_text("x = 1\n", encoding="utf-8")

    actions = [
        {
            "action": "my_action",
            "target": {"path": str(target_py)},
            "marker": str(marker_path),
        }
    ]

    from selfcoder.orchestrator import run_actions_on_files
    modified = run_actions_on_files([str(target_py)], actions, dry_run=False)

    # Even if file content is unchanged, our transformer should have run
    assert marker_path.exists(), "plugin transformer did not run"
    assert isinstance(modified, list)