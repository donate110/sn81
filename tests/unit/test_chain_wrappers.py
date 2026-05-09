"""Smoke tests for reliquary.infrastructure.chain async wrappers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_blocks_until_next_epoch_wraps_subtensor():
    """Wrapper delegates to subtensor.blocks_until_next_epoch under wait_for."""
    from reliquary.infrastructure import chain

    fake_sub = MagicMock()
    fake_sub.blocks_until_next_epoch = MagicMock(return_value=42)

    result = await chain.blocks_until_next_epoch(fake_sub, netuid=81)
    assert result == 42
    fake_sub.blocks_until_next_epoch.assert_called_once_with(81)


@pytest.mark.asyncio
async def test_blocks_until_next_epoch_timeout():
    """A hanging subtensor call surfaces as TimeoutError, not silent hang."""
    from reliquary.infrastructure import chain

    fake_sub = MagicMock()

    def _hang(*_a, **_kw):
        import time
        time.sleep(60)  # would block forever without wait_for
    fake_sub.blocks_until_next_epoch = _hang

    # Patch the timeout constant down to a tenth of a second for the test.
    original = chain.CHAIN_READ_TIMEOUT
    chain.CHAIN_READ_TIMEOUT = 0.1
    try:
        with pytest.raises(asyncio.TimeoutError):
            await chain.blocks_until_next_epoch(fake_sub, netuid=81)
    finally:
        chain.CHAIN_READ_TIMEOUT = original
