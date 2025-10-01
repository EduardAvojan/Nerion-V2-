# Core API

This document provides a reference for the core API of Nerion V2.

## `app.nerion_chat`

This module is the main entry point for the interactive chat. It contains the following functions:

*   `main()`: The main entry point for the chat application.
*   `run_self_coding_pipeline(instruction: str, speak=None, listen_once=None) -> bool`: The entry point for the self-coding functionality.

## `app.nerion_autocoder`

This module is a command-line tool for applying AST-based code transformations. It can be used to apply changes to files, including cross-file renames.

*   `main(argv: Optional[List[str]] = None) -> int`: The main entry point for the command-line tool.
