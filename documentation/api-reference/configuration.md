# Configuration

Nerion V2 is configured through a combination of environment variables and YAML files.

## Environment Variables

A `.env.example` file is provided with a list of all the available environment variables. Copy this file to `.env` and fill in the required values.

## YAML Configuration Files

### `app/settings.yaml`

This file contains the main settings for the application, such as the default provider, the request timeout, and the cost per turn.

### `config/model_catalog.yaml`

This file defines the available LLM providers and models.

### `config/intents.yaml`

This file configures the agent's intents and how they are handled.

### `config/self_mod_policy.yaml`

This file defines the policy for the agent's self-modification capabilities.

### `config/meta_policy.yaml`

This file defines the meta-policy for the Digital Physicist's learning process.

### `config/profiles.yaml`

This file defines the different profiles that can be used to configure the agent's behavior.

### `config/tools.yaml`

This file defines the tools that are available to the agent.
