# Forward PA identification methods

**PA Model → Training → PA modeling method** offers neural fitting and MP/GMP least-squares fitting. Both learn a forward model from paired training inputs and measured outputs, and predict held-out PA outputs. They do not learn a predistorter.

The compact Studio MP/GMP presets use at most 32,768 training samples for quick CPU fits. Coefficients are estimated with normalized columns and complex128 least squares. Their current NumPy implementation is not differentiable, so they cannot serve as the differentiable PA surrogate for neural DPD training.

Arena uses the validation-selected TRes-GRU PA for **APA_200MHz_b**. Its candidate selection, held-out accuracy and weights are documented in the [PA qualification record](arena-pa-qualification.md). DPD contestants all use this same frozen PA.

ILC instead learns an input waveform that makes a PA output track a target; its waveform tracking error is not a forward PA identification metric. Studio keeps this workflow under **DPD Model → ILC linearization**. It is excluded from Arena rankings and is not the source of MP/GMP Arena training targets. See the [ILC guide](../guides/ilc-dpd.md).
