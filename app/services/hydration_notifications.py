"""Optional materialization must not prevent graph writes from completing."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def notify_hydration_changed(callback, user, *, invalidate=False):
    if callback is None:
        return
    try:
        await asyncio.wait_for(callback(user, invalidate=invalidate), timeout=2)
    except Exception:
        logger.warning(
            "Hydration notification failed; graph correction remains authoritative", exc_info=True
        )
