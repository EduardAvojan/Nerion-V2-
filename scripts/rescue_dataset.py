#!/usr/bin/env python3
"""Emergency script to extract graphs from stuck dataset_builder process.

Usage:
    python3 scripts/rescue_dataset.py --pid 47891 --output rescued_dataset.pt
"""
import argparse
import gc
import os
import sys
import signal
from pathlib import Path


def inject_save_handler(pid: int, output_path: Path):
    """Inject code into running process to save graphs and exit gracefully."""

    print(f"Attempting to rescue data from PID {pid}...")
    print(f"Output will be saved to: {output_path}")

    # Send SIGUSR1 to the process - we'll modify the process to handle this
    # But first, we need to use gdb/lldb to inject a signal handler

    injection_script = f"""
import sys
import torch
import signal
from pathlib import Path

def emergency_save(signum, frame):
    print("\\n🚨 EMERGENCY SAVE TRIGGERED", file=sys.stderr)

    # Try to find the graphs variable in the current frame chain
    current_frame = frame
    graphs = None

    while current_frame is not None:
        if 'graphs' in current_frame.f_locals:
            graphs = current_frame.f_locals['graphs']
            break
        current_frame = current_frame.f_back

    if graphs:
        output_dir = Path("{output_path.parent}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path("{output_path}")

        print(f"Found {{len(graphs)}} graphs in memory", file=sys.stderr)
        print(f"Saving to {{output_file}}...", file=sys.stderr)

        try:
            torch.save({{"samples": graphs}}, output_file)
            print(f"✅ Successfully saved {{len(graphs)}} graphs!", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(f"❌ Save failed: {{e}}", file=sys.stderr)
            sys.exit(1)
    else:
        print("❌ Could not find graphs variable in stack", file=sys.stderr)
        sys.exit(1)

signal.signal(signal.SIGUSR1, emergency_save)
print("Emergency save handler installed", file=sys.stderr)
"""

    # Write injection script to temp file
    inject_file = Path("/tmp/inject_save_handler.py")
    inject_file.write_text(injection_script)

    print("\n⚠️  This requires using gdb/lldb to inject code.")
    print("On macOS, we'll use lldb...")
    print("\nExecuting injection...")

    # Use lldb to inject the Python code
    lldb_commands = f"""
process attach --pid {pid}
expression -- (void)PyRun_SimpleString("{injection_script.replace('"', '\\"').replace(os.linesep, '\\n')}")
process detach
quit
"""

    lldb_script = Path("/tmp/lldb_inject.txt")
    lldb_script.write_text(lldb_commands)

    print(f"\n📝 LLDB script written to {lldb_script}")
    print("\n⚠️  You need to run this manually with sudo:")
    print(f"    sudo lldb -s {lldb_script}")
    print(f"\nAfter injection succeeds, send SIGUSR1:")
    print(f"    kill -USR1 {pid}")

    return 1  # Manual intervention required


def main():
    parser = argparse.ArgumentParser(description="Rescue dataset from stuck process")
    parser.add_argument("--pid", type=int, required=True, help="Process ID")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")

    args = parser.parse_args()

    # Check if process exists
    try:
        os.kill(args.pid, 0)  # Signal 0 just checks if process exists
    except OSError:
        print(f"❌ Process {args.pid} does not exist or is not accessible")
        sys.exit(1)

    result = inject_save_handler(args.pid, args.output)
    sys.exit(result)


if __name__ == "__main__":
    main()
