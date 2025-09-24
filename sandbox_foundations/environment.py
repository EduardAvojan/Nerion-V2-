"""Phase 1 Environment: apply actions, run tests, and report outcomes."""

import os
import subprocess
from enum import Enum, auto


class Action(Enum):
    """Actions Nerion can perform on math_logic.py in the toy universe."""

    CHANGE_OPERATOR_ADD_TO_SUB = auto()


class ToyUniverseEnvironment:
    """Manage toy universe state transformations and verification."""

    def __init__(self):
        self.file_path = os.path.join(os.path.dirname(__file__), "math_logic.py")
        try:
            with open(self.file_path, "r", encoding="utf-8") as file_handle:
                self.original_code = file_handle.read()
        except FileNotFoundError:
            print(f"FATAL: Environment file not found at {self.file_path}")
            self.original_code = ""

    def _apply_action(self, action: Action):
        """Modify math_logic.py based on the requested action."""
        if action == Action.CHANGE_OPERATOR_ADD_TO_SUB:
            modified_code = self.original_code.replace("return a + b", "return a - b")
            with open(self.file_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(modified_code)

    def _run_tests(self) -> bool:
        """Run pytest for the toy universe and return pass/fail."""
        result = subprocess.run(
            ["pytest", "-q"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(self.file_path),
        )
        return result.returncode == 0

    def _restore_file(self):
        """Restore math_logic.py to its original state."""
        if self.original_code:
            with open(self.file_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(self.original_code)

    def step(self, action: Action) -> bool:
        """Apply an action, run tests, restore state, and report outcome."""
        print(f"Executing action: {action.name}")
        self._apply_action(action)

        outcome_is_success = self._run_tests()
        print(f"  -> Outcome: {'Tests Passed' if outcome_is_success else 'Tests FAILED'}")

        self._restore_file()
        print("  -> Environment restored to original state.")

        return outcome_is_success


def main():
    """Demonstrate environment usage with a single action."""
    env = ToyUniverseEnvironment()
    env.step(Action.CHANGE_OPERATOR_ADD_TO_SUB)


if __name__ == "__main__":
    main()
