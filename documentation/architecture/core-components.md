# Core Components

Nerion V2 is built on a set of core components that work together to provide a seamless and intelligent user experience. While the `core` directory in the codebase contains stubs for these components, their actual implementation is distributed throughout the `app` directory.

This document provides a conceptual overview of these components and their roles in the system.

## Dialog Manager

The Dialog Manager is responsible for managing the conversation flow between the user and the agent. It keeps track of the conversation history, the current context, and the user's intent.

## Intent Router

The Intent Router is responsible for determining the user's intent based on their input. It uses a combination of natural language understanding (NLU) and pattern matching to classify the user's intent and route it to the appropriate component for handling.

## Planner

The Planner is responsible for creating a plan of action to fulfill the user's request. It takes the user's intent as input and generates a sequence of steps that the agent needs to take to achieve the desired outcome. For example, if the user asks to refactor a piece of code, the Planner will generate a plan that includes steps for analyzing the code, identifying the areas for improvement, and applying the necessary changes.

## Tooling

The Tooling component provides the agent with a set of tools that it can use to interact with the outside world. These tools include the ability to:

*   Read and write files
*   Execute shell commands
*   Search the web
*   Interact with APIs

The Tooling component is designed to be extensible, allowing new tools to be added as needed.
