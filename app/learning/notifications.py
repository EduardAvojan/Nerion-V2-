from __future__ import annotations
from typing import Protocol

class Notifier(Protocol):
    def notify(self, title: str, message: str) -> None: ...

class MacNotifier:
    def notify(self, title: str, message: str) -> None:
        try:
            import subprocess
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], check=False)
        except Exception:
            pass

class LinuxNotifier:
    def notify(self, title: str, message: str) -> None:
        try:
            import subprocess
            subprocess.run(["notify-send", title, message], check=False)
        except Exception:
            pass

class WindowsNotifier:
    def notify(self, title: str, message: str) -> None:
        try:
            from win10toast import ToastNotifier  # type: ignore
            ToastNotifier().show_toast(title, message, duration=5, threaded=True)
        except Exception:
            pass

class ConsoleNotifier:
    def notify(self, title: str, message: str) -> None:
        print(f"[NOTIFY] {title}\n{message}")


def get_notifier() -> Notifier:
    import platform
    sys = platform.system().lower()
    if "darwin" in sys:
        return MacNotifier()
    if "linux" in sys:
        return LinuxNotifier()
    if "windows" in sys:
        return WindowsNotifier()
    return ConsoleNotifier()
