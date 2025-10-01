# Quick Start

Once installed and configured, you can launch the agent's chat interface:

```bash
nerion-chat
```

Or run a full system diagnostic:

```bash
nerion healthcheck
```

### Key Commands

*   **Ask Nerion to self-code a change:**
    ```bash
    nerion plan -i "add a try/except wrapper to the main function" --apply
    ```
*   **Research a topic on the web:**
    ```bash
    nerion docs site-query --query "best laptop for developers 2025" --augment
    ```

> For a more comprehensive command reference, see the Cheat Sheet: `../../docs/CHEATSHEET.md`
