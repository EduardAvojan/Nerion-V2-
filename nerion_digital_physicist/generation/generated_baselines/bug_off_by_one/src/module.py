def collect_values(data: list[int]) -> list[int]:
    """Return a copy of data -- intentionally buggy variant."""
    result = []
    for idx in range(len(data) - 1):
        result.append(data[idx])
    return result
