# Configuration

Nerion V2 is configured through a combination of environment variables and YAML files.

## Environment Variables

The primary way to configure Nerion V2 is through environment variables. You can set these variables in a `.env` file in the root of the project.

A `.env.example` file is provided with a list of all the available environment variables. Copy this file to `.env` and fill in the required values.

```bash
cp .env.example .env
```

## Configuration Files

More advanced configuration is handled through YAML files located in the `config` directory. These files allow you to customize the agent's behavior, including intents, models, and policies.

Here's a brief overview of the main configuration files:

*   `config/model_catalog.yaml`: Defines the available LLM providers and models.
*   `config/intents.yaml`: Configures the agent's intents and how they are handled.
*   `config/self_mod_policy.yaml`: Defines the policy for the agent's self-modification capabilities.

For a more detailed explanation of the configuration options, please refer to the [API Reference](./../api-reference/configuration.md) section.
