# Customizing Your Agent

This tutorial will guide you through the process of customizing your Nerion V2 agent.

## 1. Changing the Agent's Name

To change the agent's name, you can set the `NERION_AGENT_NAME` environment variable. For example:

```bash
export NERION_AGENT_NAME="My Awesome Agent"
```

## 2. Changing the Greeting Message

To change the greeting message, you can modify the `app/nerion_chat.py` file. Look for the following code:

```python
if hr < 12:
    greeting = "Good morning, I'm ready for your commands."
elif hr < 18:
    greeting = "Good afternoon, I'm ready for your commands."
else:
    greeting = "Good evening, I'm ready for your commands."
```

You can change the greeting message to whatever you want.

## 3. Adding New Commands

To add new commands, you can modify the `app/nerion_chat.py` file. Look for the `EXIT_COMMANDS`, `SLEEP_COMMANDS`, `INTERRUPT_COMMANDS`, `MUTE_COMMANDS`, and `UNMUTE_COMMANDS` sets.

You can add new commands to these sets to extend the agent's vocabulary.

## 4. Customizing the Core Components

For more advanced customization, you can modify the core components of the agent, such as the Dialog Manager, the Intent Router, and the Planner.

For more information on the core components, please refer to the [Understanding the Core Components](./core-components.md) tutorial.
