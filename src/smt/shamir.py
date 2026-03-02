"""Shamir's (n,t)-threshold secret sharing over GF(p).

Default field: Mersenne prime 2^127-1. t shares reconstruct; fewer reveal nothing.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

MERSENNE_127 = (1 << 127) - 1


@dataclass(frozen=True)
class Share:
    """A single Shamir share (x, y) where y = f(x) mod p."""

    x: int
    y: int


class ShamirSecretSharing:
    """(n, t)-threshold secret sharing over GF(p). Default prime: 2^127-1."""

    def __init__(self, prime: int = MERSENNE_127) -> None:
        self.p = prime

    def share(
        self,
        secret: int,
        n: int,
        t: int,
        evaluation_points: list[int] | None = None,
    ) -> list[Share]:
        """Split secret into n shares with threshold t. Default x-points: 1..n."""
        if not 1 <= t <= n:
            raise ValueError(f"Need 1 <= t <= n, got t={t}, n={n}")
        if not 0 <= secret < self.p:
            raise ValueError(f"Secret must be in [0, p), got {secret}")

        coeffs = self._random_polynomial(secret, t - 1)

        if evaluation_points is None:
            evaluation_points = list(range(1, n + 1))
        if len(evaluation_points) != n:
            raise ValueError(f"Need {n} evaluation points, got {len(evaluation_points)}")

        shares = []
        for x in evaluation_points:
            y = self._eval_poly(coeffs, x)
            shares.append(Share(x=x, y=y))

        return shares

    def reconstruct(self, shares: list[Share]) -> int:
        """Reconstruct secret from t+ shares via Lagrange interpolation at x=0."""
        if not shares:
            raise ValueError("Need at least one share to reconstruct")

        xs = [s.x for s in shares]
        if len(set(xs)) != len(xs):
            raise ValueError("Duplicate evaluation points in shares")

        return self._lagrange_interpolate_at_zero(shares)

    def _random_polynomial(self, constant: int, degree: int) -> list[int]:
        coeffs = [constant % self.p]
        for _ in range(degree):
            coeffs.append(secrets.randbelow(self.p))
        return coeffs

    def _eval_poly(self, coeffs: list[int], x: int) -> int:
        result = 0
        for c in reversed(coeffs):
            result = (result * x + c) % self.p
        return result

    def _lagrange_interpolate_at_zero(self, shares: list[Share]) -> int:
        """Lagrange interpolation evaluated at x = 0.

        For points (x_i, y_i), the Lagrange basis polynomial at x=0 is:
            L_i(0) = prod_{j != i} (0 - x_j) / (x_i - x_j)
                   = prod_{j != i} (-x_j) / (x_i - x_j)

        The interpolated value at 0 is sum_i y_i * L_i(0).
        """
        p = self.p
        k = len(shares)
        secret = 0

        for i in range(k):
            xi, yi = shares[i].x, shares[i].y
            numerator = 1
            denominator = 1
            for j in range(k):
                if i == j:
                    continue
                xj = shares[j].x
                numerator = (numerator * (-xj)) % p
                denominator = (denominator * (xi - xj)) % p

            lagrange_coeff = (numerator * pow(denominator, p - 2, p)) % p
            secret = (secret + yi * lagrange_coeff) % p

        return secret


def share_secret(
    secret: int,
    n: int,
    t: int,
    prime: int = MERSENNE_127,
) -> list[Share]:
    """Convenience: split secret into n shares with threshold t."""
    sss = ShamirSecretSharing(prime)
    return sss.share(secret, n, t)


def reconstruct_secret(
    shares: list[Share],
    prime: int = MERSENNE_127,
) -> int:
    """Convenience: reconstruct secret from shares."""
    sss = ShamirSecretSharing(prime)
    return sss.reconstruct(shares)


def share_bytes(
    data: bytes,
    n: int,
    t: int,
    prime: int = MERSENNE_127,
) -> tuple[list[list[Share]], int]:
    """Split bytes into share-chunks (each chunk fits GF(p)). Returns (chunks, len)."""
    chunk_size = (prime.bit_length() - 1) // 8
    if chunk_size <= 0:
        raise ValueError("Prime too small for byte-level sharing")

    sss = ShamirSecretSharing(prime)
    all_shares: list[list[Share]] = []

    for offset in range(0, len(data), chunk_size):
        chunk = data[offset : offset + chunk_size]
        secret = int.from_bytes(chunk, byteorder="big")
        if secret >= prime:
            raise ValueError("Chunk value exceeds field prime")
        shares = sss.share(secret, n, t)
        all_shares.append(shares)

    return all_shares, len(data)


def reconstruct_bytes(
    share_chunks: list[list[Share]],
    original_length: int,
    prime: int = MERSENNE_127,
) -> bytes:
    """Reconstruct bytes from share_chunks, trimmed to original_length."""
    chunk_size = (prime.bit_length() - 1) // 8
    sss = ShamirSecretSharing(prime)
    result = bytearray()

    for shares in share_chunks:
        secret = sss.reconstruct(shares)
        remaining = original_length - len(result)
        this_chunk = min(chunk_size, remaining)
        result.extend(secret.to_bytes(this_chunk, byteorder="big"))

    return bytes(result)
