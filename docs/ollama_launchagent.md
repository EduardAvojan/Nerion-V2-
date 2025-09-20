# Ollama LaunchAgent Automation (macOS)

To keep Nerion’s LLM backend ready without manual `ollama serve` commands, you
can install a user LaunchAgent that runs the Ollama daemon in the background as
soon as you log in.

```bash
# install (runs instantly and enables auto-start at login)
./scripts/setup_ollama_launchagent.sh install

# remove
./scripts/setup_ollama_launchagent.sh remove
```

## What the script does

- Creates `~/Library/LaunchAgents/io.nerion.ollama.autostart.plist` pointing to
  the `ollama serve` command.
- Registers the agent via `launchctl load -w`, so the daemon starts right away
  and at every login.
- Captures stdout/stderr to `~/Library/Logs/ollama-launchagent.log`.

## Requirements & notes

- Only available on macOS (LaunchAgents are macOS-specific).
- The script reuses whichever `ollama` binary is currently on your `PATH`.
- If you switch Ollama installations, rerun the installer so the plist picks up
  the new binary path.
- After installation you can verify the status with:

  ```bash
  launchctl print gui/$UID/io.nerion.ollama.autostart
  ```

Once the agent is active, Nerion’s Electron HOLO shell will always find the
local inference endpoint, restoring full LLM responses without extra steps.
