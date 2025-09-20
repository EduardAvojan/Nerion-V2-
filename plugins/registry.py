


class TransformerRegistry:
    def __init__(self):
        self._transformers = {}

    def add_transformer(self, name, fn):
        # Keep a local record for introspection/debugging
        self._transformers[name] = fn
        # Also register with core so actions actually execute
        try:
            from selfcoder.actions.transformers import register_external_transformer
            register_external_transformer(name, fn)
        except Exception:
            # If core is not importable here (e.g., partial env), silently keep local
            pass

    def get_all(self):
        return dict(self._transformers)


class CliRegistry:
    def __init__(self):
        self._cli_callbacks = []

    def add_cli_extension(self, callback):
        self._cli_callbacks.append(callback)

    def extend_parser(self, subparsers):
        for cb in self._cli_callbacks:
            cb(subparsers)


# Singleton instances
transformer_registry = TransformerRegistry()
cli_registry = CliRegistry()

__all__ = ["transformer_registry", "cli_registry"]