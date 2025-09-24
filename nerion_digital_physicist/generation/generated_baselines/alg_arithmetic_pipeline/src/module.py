from __future__ import annotations



def process_numbers(values: list[float]) -> list[float]:
    """Apply a deterministic arithmetic pipeline to the input list."""
    result: list[float] = []
    for item in values:
        current = float(item)
        current += 3.0
        current *= 2.0
        current = max(0.0, min(100.0, current))
        result.append(current)
    return result

def demo_input() -> list[float]:
    return [float(i) for i in range(5)]
