# MyCustomPA — dummy dataset for tutorial purpose

This is **synthetic**, generated independently of every measured PA dataset.
It demonstrates paired CSV samples, loading, plotting and a small training run.
It is not a physical PA measurement, standard-conformance capture or benchmark.

`data.csv` contains `I_in,Q_in,I_out,Q_out`: each row is one simultaneous input/output
sample. The generator puts all 64 square-QAM points on 64 active bins of each
2,560-sample IFFT frame, with no cyclic prefix, at an 80 MHz sample rate. A simple
memoryless compression/phase function produces the output. `demod.py` uses the
existing IFFT-frame receiver with matching metadata; output plots use the
existing input-referenced equalizer. Plotting recovered symbols is not an EVM test.

Reproduce the 102,400 rows with `python -m datasets.MyCustomPA.generate` from the
repository root. The fixed generator seed and signal parameters are in `spec.json`.
The packaged example retains a contiguous 60/20/20 author split with no guards.
Creating a new dataset in Studio uses the shared split protocol, including its
boundary guards; the review step shows the resulting sample counts.

In Studio, choose **Create Your Own Dataset**, upload `data.csv`, confirm the four
column roles, then choose the split percentages and **synthetic** origin. Supply
the sample rate and bandwidth when known. Alternatively, use two complex columns:

```csv
input,output
0.1+0.2j,0.17+0.34j
-0.3+0.1i,-0.50+0.17i
```

Both `i` and `j` are accepted. A complete file needs enough paired samples for
three non-empty splits and the chosen guards. Do not copy the two-row illustration
as a training capture. Unknown imported waveforms show raw I/Q until a receiver
or reference waveform has been explicitly bound.
