# OpenDPD 2.2.1 — Faster Studio training

Try [OpenDPD Studio](https://opendpd.com/studio/) or install `pip install -U "opendpd[gui]==2.2.1"`.

- Accelerate supported native CUDA models and frozen-PA cascades with graph replay, while preserving optimizer updates, precision, training data and shuffled batch order.
- Fetch in-memory IQ batches with vectorized indexing.
- Set quick trial to **10 epochs** and full training to **150 epochs**. Explicit saved configurations and published benchmark budgets retain their own epoch counts.
- Update plots **once per epoch** by reusing validation results. Advanced settings allow an explicit batch interval; the option and warning are red because frequent previews can severely slow training.
- Preserve preview cadence in exported configurations, refresh web plots on preview events, and repack copied RNN weights for optional batch previews.

A paired DPA_160MHz experiment measured **3.69× PA** and **2.28× DPD** end-to-end acceleration on the production RTX 4090 Laptop GPU container. Both versions used four epochs with identical settings; selected checkpoint tensors and final metrics were bitwise identical. These measurements do not include savings from fewer default epochs and are not a promise for every workload. See [measurement details](../performance/studio-2.2.1.md).

CUDA replay applies to reviewed GRU, DGRU, LSTM, TRes-GRU and RVTDCNN paths. Other models and unsupported execution conditions retain eager training. Existing upload validation, checkpoint downloads, isolation, queue limits and temporary-file cleanup are preserved.
