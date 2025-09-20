

from __future__ import annotations
from typing import List
import sys
import datetime as dt


def tonight_time(local_hour: int = 2) -> dt.datetime:
    """Return a datetime for the next occurrence of `local_hour:00` (typically 2am tonight)."""
    now = dt.datetime.now()
    next_day = now if now.hour < local_hour else now + dt.timedelta(days=1)
    return next_day.replace(hour=local_hour, minute=0, second=0, microsecond=0)


def build_self_learn_cmd(data_since: str = "30d", domain: str = "all", dry_run: bool = False) -> List[str]:
    """Build the command list to run the self-learn fine-tune placeholder.

    Example output:
      [sys.executable, '-m', 'selfcoder.cli_ext.self_learn', 'fine-tune', '--data-since', '30d', '--domain', 'all']
    """
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.self_learn",
        "fine-tune",
        "--data-since",
        str(data_since),
        "--domain",
        str(domain),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def advice_for_os_schedule(when: dt.datetime, cmd: List[str]) -> str:
    """Return human-friendly instructions to schedule the command at `when`.

    We do not create OS tasks here; we print guidance for macOS/Linux/Windows.
    """
    ts = when.strftime("%Y-%m-%d %H:%M")
    command = " ".join(map(str, cmd))
    return (
        f"Schedule this command at {ts}:\n\n"
        f"  {command}\n\n"
        "Examples:\n"
        "  • macOS (launchd): Use Lingon or create a LaunchAgent that runs the command at the time.\n"
        "  • Linux (cron): Run `crontab -e` and add a line for the desired minute/hour to execute the command.\n"
        "  • Windows (Task Scheduler): Create a Basic Task and point it at the command above.\n"
    )

# Helper functions to translate user choices into concrete scheduling advice

def user_choice_to_time(choice: str) -> dt.datetime:
    """Convert a user scheduling choice string into a datetime object.

    Supported choices: 'tonight', 'tomorrow', 'now', 'HH:MM', or ISO datetime string.
    """
    now = dt.datetime.now()
    lc = choice.strip().lower()
    if lc == "tonight":
        return tonight_time()
    elif lc == "tomorrow":
        tomorrow = now + dt.timedelta(days=1)
        return tomorrow.replace(hour=2, minute=0, second=0, microsecond=0)
    elif lc == "now":
        return now
    else:
        # Try to parse as HH:MM today or ISO string
        try:
            if ":" in lc and len(lc) <= 5:
                hour, minute = map(int, lc.split(":"))
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                # If that time has passed today, schedule for tomorrow
                if candidate < now:
                    candidate += dt.timedelta(days=1)
                return candidate
            else:
                # Try ISO datetime
                return dt.datetime.fromisoformat(choice)
        except Exception as e:
            raise ValueError(f"Unrecognized scheduling choice: {choice!r}") from e


def user_choices_to_advice(
    schedule_choice: str,
    data_since_choice: str = "30d",
    domain_choice: str = "all",
    dry_run: bool = False,
) -> str:
    """Given user choices, return scheduling advice string."""
    when = user_choice_to_time(schedule_choice)
    cmd = build_self_learn_cmd(data_since=data_since_choice, domain=domain_choice, dry_run=dry_run)
    return advice_for_os_schedule(when, cmd)