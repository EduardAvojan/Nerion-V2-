from __future__ import annotations

import math


class MSPRT:
    """Minimal Sequential Probability Ratio Test for Bernoulli uplift.

    Decides between arm A and B using a simple uplift hypothesis with
    symmetric log-likelihood thresholds. Returns one of: 'accept_a',
    'accept_b', or 'continue'.
    """

    def __init__(self, logA: float = 3.0, logB: float = -3.0, min_n: int = 200):
        self.A = float(logA)
        self.B = float(logB)
        self.min_n = int(min_n)

    def decide(self, s_a: int, f_a: int, s_b: int, f_b: int) -> str:
        n = int(s_a + f_a + s_b + f_b)
        if n < int(self.min_n):
            return "continue"
        # Pooled baseline rate
        p0 = (s_a + s_b) / max(1.0, float(n))
        # 10% uplift alternative (clipped to [0.001,0.999])
        p1 = min(0.999, max(0.001, p0 * 1.10))
        # Log-likelihood ratio in favor of B over A
        def _ll(s: int, f: int, p: float) -> float:
            # Avoid log(0)
            p = min(0.999999, max(0.000001, p))
            return s * math.log(p) + f * math.log(1.0 - p)

        ll_b = _ll(int(s_b), int(f_b), p1) + _ll(int(s_a), int(f_a), p0)
        ll_a = _ll(int(s_a), int(f_a), p1) + _ll(int(s_b), int(f_b), p0)
        llr = ll_b - ll_a
        if llr > self.A:
            return "accept_b"
        if llr < self.B:
            return "accept_a"
        return "continue"

