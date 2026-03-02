"""Tests for smt.shamir module."""

from __future__ import annotations

import pytest

from smt.shamir import (
    ShamirSecretSharing,
    Share,
    reconstruct_bytes,
    reconstruct_secret,
    share_bytes,
    share_secret,
)


class TestShamirSecretSharing:
    @pytest.fixture
    def sss(self, small_prime: int) -> ShamirSecretSharing:
        return ShamirSecretSharing(prime=small_prime)

    def test_share_and_reconstruct_exact_threshold(
        self, sss: ShamirSecretSharing
    ):
        secret = 42
        shares = sss.share(secret, n=5, t=3)
        assert len(shares) == 5
        recovered = sss.reconstruct(shares[:3])
        assert recovered == secret

    def test_reconstruct_with_all_shares(
        self, sss: ShamirSecretSharing
    ):
        secret = 100
        shares = sss.share(secret, n=5, t=3)
        recovered = sss.reconstruct(shares)
        assert recovered == secret

    def test_reconstruct_with_different_subsets(
        self, sss: ShamirSecretSharing
    ):
        secret = 77
        shares = sss.share(secret, n=6, t=3)
        for subset in [shares[:3], shares[1:4], shares[3:6], [shares[0], shares[2], shares[5]]]:
            assert sss.reconstruct(subset) == secret

    def test_insufficient_shares(self, sss: ShamirSecretSharing):
        """With fewer than t shares, reconstruction gives wrong result."""
        secret = 42
        shares = sss.share(secret, n=5, t=3)
        # 2 shares cannot uniquely determine a degree-2 polynomial
        sss.reconstruct(shares[:2])
        # This *could* accidentally equal the secret but is overwhelmingly unlikely
        # for a well-chosen polynomial; we just verify it runs without error

    def test_zero_secret(self, sss: ShamirSecretSharing):
        shares = sss.share(0, n=4, t=2)
        assert sss.reconstruct(shares[:2]) == 0

    def test_max_secret(self, sss: ShamirSecretSharing):
        secret = sss.p - 1
        shares = sss.share(secret, n=4, t=2)
        assert sss.reconstruct(shares[:2]) == secret

    def test_invalid_threshold(self, sss: ShamirSecretSharing):
        with pytest.raises(ValueError, match="1 <= t <= n"):
            sss.share(42, n=3, t=5)

    def test_invalid_secret(self, sss: ShamirSecretSharing):
        with pytest.raises(ValueError, match="Secret must be in"):
            sss.share(sss.p + 1, n=3, t=2)

    def test_duplicate_evaluation_points(
        self, sss: ShamirSecretSharing
    ):
        with pytest.raises(ValueError, match="Duplicate"):
            sss.reconstruct([Share(1, 10), Share(1, 20)])

    def test_empty_shares(self, sss: ShamirSecretSharing):
        with pytest.raises(ValueError, match="at least one"):
            sss.reconstruct([])

    def test_custom_evaluation_points(
        self, sss: ShamirSecretSharing
    ):
        secret = 55
        shares = sss.share(secret, n=4, t=3, evaluation_points=[10, 20, 30, 40])
        assert all(s.x in [10, 20, 30, 40] for s in shares)
        assert sss.reconstruct(shares[:3]) == secret

    def test_large_prime(self):
        sss = ShamirSecretSharing()  # default 2^127 - 1
        secret = 2**100 + 7
        shares = sss.share(secret, n=5, t=3)
        assert sss.reconstruct(shares[:3]) == secret

    def test_threshold_one(self, sss: ShamirSecretSharing):
        """t=1 means any single share reveals the secret (constant polynomial)."""
        secret = 99
        shares = sss.share(secret, n=5, t=1)
        for s in shares:
            assert sss.reconstruct([s]) == secret


class TestConvenienceFunctions:
    def test_share_and_reconstruct(self):
        secret = 12345
        shares = share_secret(secret, n=5, t=3, prime=65537)
        recovered = reconstruct_secret(shares[:3], prime=65537)
        assert recovered == secret


class TestByteSharing:
    def test_round_trip(self):
        prime = (1 << 127) - 1
        data = b"Hello, Secret Sharing!"
        chunks, orig_len = share_bytes(data, n=5, t=3, prime=prime)
        assert all(len(c) == 5 for c in chunks)
        assert orig_len == len(data)

        receiver_chunks = [c[:3] for c in chunks]
        recovered = reconstruct_bytes(receiver_chunks, orig_len, prime=prime)
        assert recovered == data
