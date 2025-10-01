# Self-Coding Engine

Nerion V2's self-coding engine is a powerful feature that allows the agent to modify its own codebase. This engine is powered by a combination of natural language understanding, code analysis, and automated code transformations.

## `nerion_autocoder.py`

The `nerion_autocoder.py` script is a command-line tool that provides an interface to the self-coding engine. It allows you to apply AST-based code transformations to your files. You can use it to perform tasks such as:

*   Adding logging to functions
*   Normalizing imports
*   Renaming symbols across files

## AST-Based Transformations

The self-coding engine uses Abstract Syntax Trees (ASTs) to represent and manipulate code. This allows the agent to make precise and safe changes to the codebase, without the risk of introducing syntax errors.

The engine includes a library of AST-based transformations that can be used to perform common refactoring tasks. These transformations are designed to be composable, allowing the agent to combine them to perform more complex changes.

## The Self-Coding Pipeline

The self-coding pipeline is the process that the agent follows to modify its own code. It consists of the following steps:

1.  **Instruction:** The user provides an instruction in natural language, such as "add a try/except wrapper to the main function".
2.  **Planning:** The Planner creates a plan of action to fulfill the user's request.
3.  **Execution:** The agent executes the plan, using the `nerion_autocoder.py` script and the AST-based transformations to modify the code.
4.  **Verification:** The agent runs tests to ensure that the changes have not introduced any regressions.
5.  **Commit:** If the tests pass, the agent commits the changes to the codebase.
