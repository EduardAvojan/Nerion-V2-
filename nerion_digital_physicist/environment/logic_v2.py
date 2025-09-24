"""Phase 2 logic module introducing scope-sensitive constructs."""

# A global variable that should remain untouched by local renames.
value = 100


def add_scoped(value: int, increment: int) -> int:
    """Return the sum of the local value and increment."""
    result = value + increment
    return result


def subtract_scoped(value: int, decrement: int) -> int:
    """Return the difference between the local value and decrement."""
    result = value - decrement
    return result


def multiply_scoped(value: int, multiplier: int) -> int:
    """Return ``value`` multiplied by ``multiplier``.

    The implementation should raise ``ValueError`` when ``multiplier`` is negative
    so the calling code can surface invalid scaling attempts gracefully.
    """
    raise NotImplementedError(
        "Generative action IMPLEMENT_DOCSTRING_FUNCTION must provide the body."
    )
