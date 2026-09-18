# OpenDPD Studio 2.2.15

The **Output vs. reference** section in PA and DPD training now shows two
waveform charts: **I (in-phase)** and **Q (quadrature)**. Each compares only
the model output with its reference, using a solid prediction and dashed
reference. Both components start with the same amplitude scale and sample
range. Legends sit above the axes in larger text.

The initial view shows the first 128 samples. Choose **Full excerpt** to
see the entire available waveform, or zoom and pan. This changes the view,
not the samples. Choose an individual signal to inspect the original input,
predistorted input or a without-DPD baseline without overlaying unrelated
signals. The same controls are available on completed result pages.

Frequency plots retain their separate signal-chain positions and now use
the full section width during training. Live waveform labels carry the
same measured/synthetic/model source information as their spectra.

Training, checkpoint selection and metric calculations are unchanged.
The valid-sample ACLR protocol, TRes-GRU and batch size 16 from
[2.2.14](release-notes-2.2.14.md) remain the defaults.
