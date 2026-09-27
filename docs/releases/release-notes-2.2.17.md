# OpenDPD Studio 2.2.17

Addresses [Issue #59](https://github.com/lab-emi/OpenDPD/issues/59): use your own generated waveform as the input to a Virtual PA.

**Import custom signal** is available in Signal Generator and PA Library, locally and in hosted Studio. Upload a CSV with I/Q columns, a complex column or a real column; select the columns and enter the actual sample rate and bandwidth. **Import & use in Virtual PA** selects the saved input, ready for PA simulation and automatic paired-dataset creation.

The complete capture is converted to the framework's float32 I/Q format. There is no normalization, filtering, resampling or truncation to the Analyzer's preview window. Inputs retain their CSV hash, column selection, sampling metadata and sample hash. Uploaded inputs remain marked as uploaded; PA outputs and paired training datasets are synthetic. Input CSV/ZIP exports carry the metadata, and the paired dataset supports the existing PA/DPD training and testing workflow.

Imports accept 256–1,000,000 samples in a CSV up to 25 MiB. PA simulation with a training dataset needs at least 8,192 samples. Invalid numbers, column selections, missing sources and changed stored samples are rejected. Hosted imports use the existing private session, rate and storage limits.

Verification covers CSV formats, complete captures beyond 262,144 samples, export round trips, PA simulation, real CPU PA/DPD training and testing, authenticated API access, hosted isolation and quotas, and frontend interaction tests. The browser verification script exercises desktop and mobile against a real local API. These checks cover software PA models; physical RF measurements are outside this feature.

[Open Studio](https://opendpd.com/studio/) · [Import guide](../guides/signal-generator.md#import-your-own-waveform) · [Virtual PA Library](../guides/virtual-pa-library.md)
