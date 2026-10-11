"""Spike stand-in for the proposed binary endpoints. NOT part of OpenDPD: a throwaway server that measures the MATLAB client.

usage: python standin_server.py PORT CONTROL_DIR
  POST /infer   body: an .npy (float32 N x 2); response: an .npy of the same shape, y = 0.5 * x  (header x-samples: N)
  GET  /big?mb= response: that many MiB of bytes, streamed
  POST /sink    body: any bytes; response: JSON {bytes, sha256}
Stops when CONTROL_DIR/standin-stop appears.
"""
import asyncio
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

BLOCK = bytes(range(256)) * 4096          # 1 MiB


async def infer(request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
    x = np.load(io.BytesIO(bytes(body)), allow_pickle=False)
    y = (x * np.float32(0.5)).astype("<f4")
    out = io.BytesIO()
    np.save(out, y, allow_pickle=False)
    return Response(out.getvalue(), media_type="application/x-npy", headers={"x-samples": str(len(x))})


async def big(request):
    mb = int(request.query_params.get("mb", "1"))

    async def blocks():
        for _ in range(mb):
            yield BLOCK

    return StreamingResponse(blocks(), media_type="application/octet-stream", headers={"content-length": str(mb << 20)})


async def sink(request):
    digest, total = hashlib.sha256(), 0
    async for chunk in request.stream():
        digest.update(chunk)
        total += len(chunk)
    return JSONResponse({"bytes": total, "sha256": digest.hexdigest()})


app = Starlette(routes=[Route("/infer", infer, methods=["POST"]), Route("/big", big), Route("/sink", sink, methods=["POST"])])


async def main(port, control):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    stop = Path(control) / "standin-stop"
    while not stop.exists() and not task.done():
        await asyncio.sleep(0.2)
    server.should_exit = True
    await task


asyncio.run(main(int(sys.argv[1]), sys.argv[2]))
