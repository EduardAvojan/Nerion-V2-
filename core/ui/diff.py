from typing import Iterable, Tuple

def render_unified(lines: Iterable[Tuple[str, str]]) -> str:
    """
    Render a minimal colored diff-like output.
    lines: iterable of (tag, text) where tag in {'+', '-', ' '} (add, del, context).
    """
    out = []
    for tag, text in lines:
        if tag == '+':
            out.append(f"\x1b[32m+ {text}\x1b[0m")
        elif tag == '-':
            out.append(f"\x1b[31m- {text}\x1b[0m")
        else:
            out.append(f"  {text}")
    return "\n".join(out)
