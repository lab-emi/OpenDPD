"""Cold-cache training probe for the hosted GPU image and its job sandbox."""
import warnings

import torch

from backbones.tres_deltagru import DeltaGRULayer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the GPU container")
    torch.manual_seed(0)
    model = DeltaGRULayer(6, 15, 1).cuda()
    model.debug = 0
    features = torch.randn(2, 17, 6, device="cuda", requires_grad=True)
    if not model._can_use_triton(features):
        raise RuntimeError("The hosted GPU image must support fused DeltaGRU")
    # Hosted jobs must keep acceleration; a portable fallback is not a pass.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        output = model(features)
        output.square().mean().backward()
    tensors = [output, features.grad, *(p.grad for p in model.parameters())]
    if any(value is None or not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("DeltaGRU CUDA forward/backward probe failed")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
