"""Fused CUDA implementation of the DeltaGRU recurrence.

The public helper in this module is intentionally small and optional.  Importing
``backbones.tres_deltagru`` must continue to work on machines without Triton;
callers should check :func:`can_use_triton_deltagru` before dispatching here.

The kernels preserve the eager recurrence, including its threshold comparisons
and separate input/hidden accumulator states.  The backward kernel performs the
reverse recurrence and leaves the two weight-gradient reductions to cuBLAS.
That avoids a highly contended atomic reduction over batch and time.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except (ImportError, OSError):  # pragma: no cover - CPU-only installations.
    triton = None
    tl = None
    libdevice = None


def can_use_triton_deltagru(input: torch.Tensor, hidden_size: int) -> bool:
    """Return whether the fused kernel supports this tensor configuration."""

    return (
        triton is not None
        and input.ndim == 3
        and input.shape[0] > 0
        and input.shape[1] > 0
        and input.is_cuda
        and input.is_contiguous()
        and torch.version.cuda is not None
        and getattr(torch.version, "hip", None) is None
        and input.dtype == torch.float32
        and 0 < input.shape[-1] <= 16
        and 0 < hidden_size <= 32
    )


if triton is not None:

    @triton.jit
    def _deltagru_forward_kernel(
        x_ptr,
        x_p0_ptr,
        h0_ptr,
        h_p0_ptr,
        dm_nh0_ptr,
        dm0_ptr,
        weight_x_ptr,
        weight_h_ptr,
        out_ptr,
        delta_x_ptr,
        delta_h_ptr,
        gate_r_ptr,
        gate_z_ptr,
        gate_n_ptr,
        dm_nh_hist_ptr,
        mask_x_ptr,
        mask_h_ptr,
        stats_ptr,
        seq_len,
        input_size: tl.constexpr,
        hidden_size: tl.constexpr,
        th_x: tl.constexpr,
        th_h: tl.constexpr,
        collect_stats: tl.constexpr,
        save_intermediates: tl.constexpr,
        block_i: tl.constexpr,
        block_h: tl.constexpr,
    ):
        batch = tl.program_id(0)
        offs_i = tl.arange(0, block_i)
        offs_h = tl.arange(0, block_h)
        mask_i = offs_i < input_size
        mask_h = offs_h < hidden_size

        # Every program owns one batch row, so all recurrent state stays local.
        x_p = tl.load(x_p0_ptr + batch * input_size + offs_i, mask=mask_i, other=0.0)
        h = tl.load(h0_ptr + batch * hidden_size + offs_h, mask=mask_h, other=0.0)
        h_p = tl.load(h_p0_ptr + batch * hidden_size + offs_h, mask=mask_h, other=0.0)
        dm_r = tl.load(dm0_ptr + batch * (3 * hidden_size) + offs_h, mask=mask_h, other=0.0)
        dm_z = tl.load(
            dm0_ptr + batch * (3 * hidden_size) + hidden_size + offs_h,
            mask=mask_h,
            other=0.0,
        )
        dm_n = tl.load(
            dm0_ptr + batch * (3 * hidden_size) + 2 * hidden_size + offs_h,
            mask=mask_h,
            other=0.0,
        )
        dm_nh = tl.load(dm_nh0_ptr + batch * hidden_size + offs_h, mask=mask_h, other=0.0)

        # The weights are loop-invariant.  Loading them before the scan also
        # gives the compiler a chance to keep this tiny model in registers.
        wx_offsets = offs_h[:, None] * input_size + offs_i[None, :]
        wh_offsets = offs_h[:, None] * hidden_size + offs_h[None, :]
        wx_r = tl.load(weight_x_ptr + wx_offsets, mask=mask_h[:, None] & mask_i[None, :], other=0.0)
        wx_z = tl.load(
            weight_x_ptr + hidden_size * input_size + wx_offsets,
            mask=mask_h[:, None] & mask_i[None, :],
            other=0.0,
        )
        wx_n = tl.load(
            weight_x_ptr + 2 * hidden_size * input_size + wx_offsets,
            mask=mask_h[:, None] & mask_i[None, :],
            other=0.0,
        )
        wh_r = tl.load(weight_h_ptr + wh_offsets, mask=mask_h[:, None] & mask_h[None, :], other=0.0)
        wh_z = tl.load(
            weight_h_ptr + hidden_size * hidden_size + wh_offsets,
            mask=mask_h[:, None] & mask_h[None, :],
            other=0.0,
        )
        wh_n = tl.load(
            weight_h_ptr + 2 * hidden_size * hidden_size + wh_offsets,
            mask=mask_h[:, None] & mask_h[None, :],
            other=0.0,
        )

        zeros_x = tl.zeros((), tl.int64)
        zeros_h = tl.zeros((), tl.int64)

        for time in tl.range(0, seq_len, loop_unroll_factor=1):
            x_offsets = (batch * seq_len + time) * input_size + offs_i
            xt = tl.load(x_ptr + x_offsets, mask=mask_i, other=0.0)

            raw_delta_x = xt - x_p
            raw_delta_h = h - h_p
            abs_delta_x = tl.abs(raw_delta_x)
            abs_delta_h = tl.abs(raw_delta_h)

            # masked_fill(abs(delta) < threshold, 0) and where(abs(delta) >=
            # threshold, current, previous) are deliberately kept separate:
            # their behaviour differs for NaNs in the eager implementation.
            keep_delta_x = ~(abs_delta_x < th_x)
            keep_delta_h = ~(abs_delta_h < th_h)
            update_x = abs_delta_x >= th_x
            update_h = abs_delta_h >= th_h
            dx = tl.where(keep_delta_x, raw_delta_x, 0.0)
            dh = tl.where(keep_delta_h, raw_delta_h, 0.0)
            x_p = tl.where(update_x, xt, x_p)
            h_p = tl.where(update_h, h, h_p)

            input_mac_r = tl.sum(wx_r * dx[None, :], axis=1) + dm_r
            input_mac_z = tl.sum(wx_z * dx[None, :], axis=1) + dm_z
            input_mac_n = tl.sum(wx_n * dx[None, :], axis=1) + dm_n
            hidden_mac_r = tl.sum(wh_r * dh[None, :], axis=1)
            hidden_mac_z = tl.sum(wh_z * dh[None, :], axis=1)
            hidden_mac_n = tl.sum(wh_n * dh[None, :], axis=1)

            dm_r = input_mac_r + hidden_mac_r
            dm_z = input_mac_z + hidden_mac_z
            dm_n = input_mac_n
            dm_nh = hidden_mac_n + dm_nh

            r = tl.sigmoid(dm_r)
            z = tl.sigmoid(dm_z)
            n = libdevice.tanh(dm_n + r * dm_nh)
            h = (1.0 - z) * n + z * h

            h_offsets = (batch * seq_len + time) * hidden_size + offs_h
            tl.store(out_ptr + h_offsets, h, mask=mask_h)
            if save_intermediates:
                tl.store(delta_x_ptr + x_offsets, dx, mask=mask_i)
                tl.store(delta_h_ptr + h_offsets, dh, mask=mask_h)
                tl.store(gate_r_ptr + h_offsets, r, mask=mask_h)
                tl.store(gate_z_ptr + h_offsets, z, mask=mask_h)
                tl.store(gate_n_ptr + h_offsets, n, mask=mask_h)
                tl.store(dm_nh_hist_ptr + h_offsets, dm_nh, mask=mask_h)
                tl.store(mask_x_ptr + x_offsets, update_x, mask=mask_i)
                tl.store(mask_h_ptr + h_offsets, update_h, mask=mask_h)

            if collect_stats:
                zeros_x += tl.sum(((dx == 0.0) & mask_i).to(tl.int64), axis=0)
                zeros_h += tl.sum(((dh == 0.0) & mask_h).to(tl.int64), axis=0)

        if collect_stats:
            tl.atomic_add(stats_ptr, zeros_x)
            tl.atomic_add(stats_ptr + 1, zeros_h)

    @triton.jit
    def _deltagru_backward_kernel(
        grad_out_ptr,
        h0_ptr,
        out_ptr,
        delta_x_ptr,
        delta_h_ptr,
        gate_r_ptr,
        gate_z_ptr,
        gate_n_ptr,
        dm_nh_hist_ptr,
        mask_x_ptr,
        mask_h_ptr,
        weight_x_ptr,
        weight_h_ptr,
        grad_x_ptr,
        grad_x_p0_ptr,
        grad_h0_ptr,
        grad_h_p0_ptr,
        grad_dm_nh0_ptr,
        grad_dm0_ptr,
        grad_mac_x_ptr,
        grad_mac_h_ptr,
        seq_len,
        input_size: tl.constexpr,
        hidden_size: tl.constexpr,
        block_i: tl.constexpr,
        block_h: tl.constexpr,
    ):
        batch = tl.program_id(0)
        offs_i = tl.arange(0, block_i)
        offs_h = tl.arange(0, block_h)
        mask_i = offs_i < input_size
        mask_h = offs_h < hidden_size

        wx_offsets = offs_h[:, None] * input_size + offs_i[None, :]
        wh_offsets = offs_h[:, None] * hidden_size + offs_h[None, :]
        wx_r = tl.load(weight_x_ptr + wx_offsets, mask=mask_h[:, None] & mask_i[None, :], other=0.0)
        wx_z = tl.load(
            weight_x_ptr + hidden_size * input_size + wx_offsets,
            mask=mask_h[:, None] & mask_i[None, :],
            other=0.0,
        )
        wx_n = tl.load(
            weight_x_ptr + 2 * hidden_size * input_size + wx_offsets,
            mask=mask_h[:, None] & mask_i[None, :],
            other=0.0,
        )
        wh_r = tl.load(weight_h_ptr + wh_offsets, mask=mask_h[:, None] & mask_h[None, :], other=0.0)
        wh_z = tl.load(
            weight_h_ptr + hidden_size * hidden_size + wh_offsets,
            mask=mask_h[:, None] & mask_h[None, :],
            other=0.0,
        )
        wh_n = tl.load(
            weight_h_ptr + 2 * hidden_size * hidden_size + wh_offsets,
            mask=mask_h[:, None] & mask_h[None, :],
            other=0.0,
        )

        grad_x_p = tl.zeros((block_i,), tl.float32)
        grad_h = tl.zeros((block_h,), tl.float32)
        grad_h_p = tl.zeros((block_h,), tl.float32)
        grad_dm_r = tl.zeros((block_h,), tl.float32)
        grad_dm_z = tl.zeros((block_h,), tl.float32)
        grad_dm_n = tl.zeros((block_h,), tl.float32)
        grad_dm_nh = tl.zeros((block_h,), tl.float32)
        h0 = tl.load(h0_ptr + batch * hidden_size + offs_h, mask=mask_h, other=0.0)

        for reverse_time in tl.range(0, seq_len, loop_unroll_factor=1):
            time = seq_len - 1 - reverse_time
            h_offsets = (batch * seq_len + time) * hidden_size + offs_h
            x_offsets = (batch * seq_len + time) * input_size + offs_i

            grad_h += tl.load(grad_out_ptr + h_offsets, mask=mask_h, other=0.0)
            h_prev = tl.load(
                out_ptr + h_offsets - hidden_size,
                mask=mask_h & (time > 0),
                other=0.0,
            )
            h_prev = tl.where(time > 0, h_prev, h0)
            r = tl.load(gate_r_ptr + h_offsets, mask=mask_h, other=0.0)
            z = tl.load(gate_z_ptr + h_offsets, mask=mask_h, other=0.0)
            n = tl.load(gate_n_ptr + h_offsets, mask=mask_h, other=0.0)
            dm_nh = tl.load(dm_nh_hist_ptr + h_offsets, mask=mask_h, other=0.0)

            grad_n = grad_h * (1.0 - z)
            grad_z = grad_h * (h_prev - n)
            grad_h_prev = grad_h * z
            grad_pre_n = grad_n * (1.0 - n * n)
            grad_r = grad_pre_n * dm_nh
            grad_dm_nh += grad_pre_n * r
            grad_dm_r += grad_r * r * (1.0 - r)
            grad_dm_z += grad_z * z * (1.0 - z)
            grad_dm_n += grad_pre_n

            grad_mac_x_r = grad_dm_r
            grad_mac_x_z = grad_dm_z
            grad_mac_x_n = grad_dm_n
            grad_mac_h_r = grad_dm_r
            grad_mac_h_z = grad_dm_z
            grad_mac_h_n = grad_dm_nh

            grad_dx = (
                tl.sum(wx_r * grad_mac_x_r[:, None], axis=0)
                + tl.sum(wx_z * grad_mac_x_z[:, None], axis=0)
                + tl.sum(wx_n * grad_mac_x_n[:, None], axis=0)
            )
            grad_dh = (
                tl.sum(wh_r * grad_mac_h_r[:, None], axis=0)
                + tl.sum(wh_z * grad_mac_h_z[:, None], axis=0)
                + tl.sum(wh_n * grad_mac_h_n[:, None], axis=0)
            )

            update_x = tl.load(mask_x_ptr + x_offsets, mask=mask_i, other=0).to(tl.int1)
            update_h = tl.load(mask_h_ptr + h_offsets, mask=mask_h, other=0).to(tl.int1)
            saved_delta_x = tl.load(delta_x_ptr + x_offsets, mask=mask_i, other=0.0)
            saved_delta_h = tl.load(delta_h_ptr + h_offsets, mask=mask_h, other=0.0)
            # For non-NaN thresholds, masked-fill's keep condition differs from
            # the state-update condition only when a raw delta is NaN.
            keep_delta_x = update_x | (saved_delta_x != saved_delta_x)
            keep_delta_h = update_h | (saved_delta_h != saved_delta_h)
            grad_xt = tl.where(keep_delta_x, grad_dx, 0.0) + tl.where(
                update_x, grad_x_p, 0.0
            )
            grad_x_p = tl.where(keep_delta_x, -grad_dx, 0.0) + tl.where(
                update_x, 0.0, grad_x_p
            )
            grad_h_prev += tl.where(keep_delta_h, grad_dh, 0.0) + tl.where(
                update_h, grad_h_p, 0.0
            )
            grad_h_p = tl.where(keep_delta_h, -grad_dh, 0.0) + tl.where(
                update_h, 0.0, grad_h_p
            )

            tl.store(grad_x_ptr + x_offsets, grad_xt, mask=mask_i)
            mac_offsets = (batch * seq_len + time) * (3 * hidden_size) + offs_h
            tl.store(grad_mac_x_ptr + mac_offsets, grad_mac_x_r, mask=mask_h)
            tl.store(
                grad_mac_x_ptr + mac_offsets + hidden_size,
                grad_mac_x_z,
                mask=mask_h,
            )
            tl.store(
                grad_mac_x_ptr + mac_offsets + 2 * hidden_size,
                grad_mac_x_n,
                mask=mask_h,
            )
            tl.store(grad_mac_h_ptr + mac_offsets, grad_mac_h_r, mask=mask_h)
            tl.store(
                grad_mac_h_ptr + mac_offsets + hidden_size,
                grad_mac_h_z,
                mask=mask_h,
            )
            tl.store(
                grad_mac_h_ptr + mac_offsets + 2 * hidden_size,
                grad_mac_h_n,
                mask=mask_h,
            )

            # Gradients for the accumulator carries at the previous timestep.
            grad_dm_r = grad_mac_x_r
            grad_dm_z = grad_mac_x_z
            grad_dm_n = grad_mac_x_n
            grad_dm_nh = grad_mac_h_n
            grad_h = grad_h_prev

        tl.store(grad_x_p0_ptr + batch * input_size + offs_i, grad_x_p, mask=mask_i)
        tl.store(grad_h0_ptr + batch * hidden_size + offs_h, grad_h, mask=mask_h)
        tl.store(grad_h_p0_ptr + batch * hidden_size + offs_h, grad_h_p, mask=mask_h)
        tl.store(grad_dm_nh0_ptr + batch * hidden_size + offs_h, grad_dm_nh, mask=mask_h)
        dm0_offsets = batch * (3 * hidden_size) + offs_h
        tl.store(grad_dm0_ptr + dm0_offsets, grad_dm_r, mask=mask_h)
        tl.store(grad_dm0_ptr + dm0_offsets + hidden_size, grad_dm_z, mask=mask_h)
        tl.store(grad_dm0_ptr + dm0_offsets + 2 * hidden_size, grad_dm_n, mask=mask_h)


class _TritonDeltaGRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        x_p0: torch.Tensor,
        h0: torch.Tensor,
        h_p0: torch.Tensor,
        dm_nh0: torch.Tensor,
        dm0: torch.Tensor,
        weight_x: torch.Tensor,
        weight_h: torch.Tensor,
        stats: torch.Tensor,
        th_x: float,
        th_h: float,
        collect_stats: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, input_size = input.shape
        hidden_size = h0.shape[-1]
        output = input.new_empty((batch_size, seq_len, hidden_size))
        delta_x = torch.empty_like(input)
        state_shape = (batch_size, seq_len, hidden_size)
        delta_h = input.new_empty(state_shape)
        gate_r = input.new_empty(state_shape)
        gate_z = input.new_empty(state_shape)
        gate_n = input.new_empty(state_shape)
        dm_nh_hist = input.new_empty(state_shape)
        mask_x = torch.empty_like(input, dtype=torch.uint8)
        mask_h = torch.empty(state_shape, device=input.device, dtype=torch.uint8)

        block_i = triton.next_power_of_2(input_size)
        block_h = triton.next_power_of_2(hidden_size)
        _deltagru_forward_kernel[(batch_size,)](
            input,
            x_p0,
            h0,
            h_p0,
            dm_nh0,
            dm0,
            weight_x,
            weight_h,
            output,
            delta_x,
            delta_h,
            gate_r,
            gate_z,
            gate_n,
            dm_nh_hist,
            mask_x,
            mask_h,
            stats,
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            th_x=th_x,
            th_h=th_h,
            collect_stats=collect_stats,
            save_intermediates=True,
            block_i=block_i,
            block_h=block_h,
            num_warps=1,
        )
        ctx.save_for_backward(
            h0,
            output,
            delta_x,
            delta_h,
            gate_r,
            gate_z,
            gate_n,
            dm_nh_hist,
            mask_x,
            mask_h,
            weight_x,
            weight_h,
        )
        ctx.input_size = input_size
        ctx.hidden_size = hidden_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "fused DeltaGRU supports first-order training only; set "
                "OPENDPD_DISABLE_TRITON_DELTAGRU=1 for higher-order autograd"
            )
        (
            h0,
            output,
            delta_x,
            delta_h,
            gate_r,
            gate_z,
            gate_n,
            dm_nh_hist,
            mask_x,
            mask_h,
            weight_x,
            weight_h,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        batch_size, seq_len, hidden_size = output.shape
        input_size = ctx.input_size

        grad_input = delta_x.new_empty(delta_x.shape)
        grad_x_p0 = delta_x.new_empty((batch_size, input_size))
        grad_h0 = output.new_empty((batch_size, hidden_size))
        grad_h_p0 = output.new_empty((batch_size, hidden_size))
        grad_dm_nh0 = output.new_empty((batch_size, hidden_size))
        grad_dm0 = output.new_empty((batch_size, 3 * hidden_size))
        grad_mac_x = output.new_empty((batch_size, seq_len, 3 * hidden_size))
        grad_mac_h = output.new_empty((batch_size, seq_len, 3 * hidden_size))

        block_i = triton.next_power_of_2(input_size)
        block_h = triton.next_power_of_2(hidden_size)
        _deltagru_backward_kernel[(batch_size,)](
            grad_output,
            h0,
            output,
            delta_x,
            delta_h,
            gate_r,
            gate_z,
            gate_n,
            dm_nh_hist,
            mask_x,
            mask_h,
            weight_x,
            weight_h,
            grad_input,
            grad_x_p0,
            grad_h0,
            grad_h_p0,
            grad_dm_nh0,
            grad_dm0,
            grad_mac_x,
            grad_mac_h,
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            block_i=block_i,
            block_h=block_h,
            num_warps=1,
        )

        flat_delta_x = delta_x.reshape(-1, input_size)
        flat_delta_h = delta_h.reshape(-1, hidden_size)
        flat_grad_mac_x = grad_mac_x.reshape(-1, 3 * hidden_size)
        flat_grad_mac_h = grad_mac_h.reshape(-1, 3 * hidden_size)
        grad_weight_x = flat_grad_mac_x.t().matmul(flat_delta_x)
        grad_weight_h = flat_grad_mac_h.t().matmul(flat_delta_h)

        return (
            grad_input,
            grad_x_p0,
            grad_h0,
            grad_h_p0,
            grad_dm_nh0,
            grad_dm0,
            grad_weight_x,
            grad_weight_h,
            None,
            None,
            None,
            None,
        )


def triton_deltagru(
    input: torch.Tensor,
    x_p0: torch.Tensor,
    h0: torch.Tensor,
    h_p0: torch.Tensor,
    dm_nh0: torch.Tensor,
    dm0: torch.Tensor,
    weight_x: torch.Tensor,
    weight_h: torch.Tensor,
    stats: torch.Tensor,
    th_x: float,
    th_h: float,
    collect_stats: bool = False,
) -> torch.Tensor:
    """Run the fused recurrence on a contiguous ``(batch, time, input)`` tensor."""

    if triton is None:  # pragma: no cover
        raise RuntimeError("Triton is not installed")
    if math.isnan(th_x) or math.isnan(th_h):
        raise ValueError("DeltaGRU thresholds must not be NaN")
    batch_size, seq_len, input_size = input.shape
    hidden_size = h0.shape[-1] if h0.ndim else 0
    expected = (
        (x_p0, (batch_size, input_size)),
        (h0, (batch_size, hidden_size)),
        (h_p0, (batch_size, hidden_size)),
        (dm_nh0, (batch_size, hidden_size)),
        (dm0, (batch_size, 3 * hidden_size)),
        (weight_x, (3 * hidden_size, input_size)),
        (weight_h, (3 * hidden_size, hidden_size)),
    )
    if not can_use_triton_deltagru(input, hidden_size):
        raise ValueError("unsupported DeltaGRU input tensor or hidden size")
    if not all(
        tensor.shape == shape
        and tensor.device == input.device
        and tensor.dtype == input.dtype
        and tensor.is_contiguous()
        for tensor, shape in expected
    ):
        raise ValueError("DeltaGRU states and weights must match input shape/device/dtype")
    if (
        stats.device != input.device
        or stats.dtype != torch.int64
        or stats.numel() < 2
        or not stats.is_contiguous()
    ):
        raise ValueError("stats must be a contiguous CUDA int64 tensor with at least 2 elements")
    if not torch.is_grad_enabled():
        output = input.new_empty((batch_size, seq_len, hidden_size))
        dummy = input.new_empty(1)
        dummy_mask = torch.empty(1, device=input.device, dtype=torch.uint8)
        _deltagru_forward_kernel[(batch_size,)](
            input,
            x_p0,
            h0,
            h_p0,
            dm_nh0,
            dm0,
            weight_x,
            weight_h,
            output,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy_mask,
            dummy_mask,
            stats,
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            th_x=th_x,
            th_h=th_h,
            collect_stats=collect_stats,
            save_intermediates=False,
            block_i=triton.next_power_of_2(input_size),
            block_h=triton.next_power_of_2(hidden_size),
            num_warps=1,
        )
        return output
    return _TritonDeltaGRUFunction.apply(
        input,
        x_p0,
        h0,
        h_p0,
        dm_nh0,
        dm0,
        weight_x,
        weight_h,
        stats,
        th_x,
        th_h,
        collect_stats,
    )
