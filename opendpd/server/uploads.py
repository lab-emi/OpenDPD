"""Run upload consumers off the event loop and keep their file alive until completion."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import TypeVar

from fastapi import UploadFile

T = TypeVar('T')


async def consume_upload(file: UploadFile, consumer: Callable[[Iterator[bytes]], T]) -> T:
    def chunks():
        while block := file.file.read(1 << 20):
            yield block

    task = asyncio.create_task(asyncio.to_thread(consumer, chunks()))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling an await cannot cancel a thread. Do not close its file or
        # remove quarantine storage while the receiver still owns it.
        try:
            await task
        finally:
            raise
    finally:
        await file.close()
