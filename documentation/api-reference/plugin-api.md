# Plugin API

Nerion V2 has a powerful plugin system that allows you to extend its functionality. You can create your own plugins to add new AST transformers and CLI commands.

## Creating a Plugin

To create a plugin, you need to create a new subdirectory in the `plugins` directory. This subdirectory must contain a `plugin.py` file.

### The `plugin.py` File

The `plugin.py` file must contain a `Plugin` class. This class can have two methods:

*   `register_transformers(transformer_registry)`: This method is used to register new AST transformers. The `transformer_registry` is an instance of the `TransformerRegistry` class, which has an `add_transformer(name, fn)` method.
*   `register_cli(cli_registry)`: This method is used to register new CLI commands. The `cli_registry` is an instance of the `CliRegistry` class, which has an `add_cli_extension(callback)` method.

### Example

Here is an example of a simple plugin that adds a new AST transformer and a new CLI command:

```python
from selfcoder.actions.transformers import register_external_transformer

class Plugin:
    def register_transformers(self, transformer_registry):
        def my_transformer(source):
            # Your transformer logic here
            return source

        transformer_registry.add_transformer("my_transformer", my_transformer)

    def register_cli(self, cli_registry):
        def my_command(subparsers):
            parser = subparsers.add_parser("my-command")
            parser.set_defaults(func=lambda args: print("Hello from my command!"))

        cli_registry.add_cli_extension(my_command)
```

## Plugin Security

To ensure the security of the system, all plugins must be explicitly allowed in the `plugins/allowlist.json` file.
