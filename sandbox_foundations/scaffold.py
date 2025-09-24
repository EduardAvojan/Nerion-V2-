"""Phase 1 Causal Scaffold: AST extraction for the toy universe."""

import ast
import os


def build_ast_from_file(file_path: str):
    """Read Python source and produce an AST module tree."""
    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            source_code = file_handle.read()
        return ast.parse(source_code)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as exc:  # noqa: BLE001 - broad for diagnostic logging
        print(f"An error occurred: {exc}")
        return None


def main():
    """Entry point to build and dump the AST for the toy universe."""
    file_path = os.path.join(os.path.dirname(__file__), "math_logic.py")

    print(f"Building AST for: {file_path}\n")

    code_ast = build_ast_from_file(file_path)

    if code_ast:
        print("--- AST Dump ---")
        print(ast.dump(code_ast, indent=4))
        print("\n--- End AST Dump ---")


if __name__ == "__main__":
    main()
