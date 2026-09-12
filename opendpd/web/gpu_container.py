"""Fixed container entrypoint: reviewed OpenDPD only, bounded allocator, no shell."""
import torch
from opendpd.runtime.worker import main

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the GPU container")
    # Cooperative allocator ceiling for the reviewed models; the GPU is not MIG-partitioned.
    # The host scheduler additionally serializes jobs and reserves desktop headroom.
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    raise SystemExit(main())
