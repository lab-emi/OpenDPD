# Reproducing published results

Run the scripts below from the repository root after [source installation](install.md). These reproduce specific paper settings and may launch multiple long training jobs; they are different from the short Studio smoke recipes. Inspect each script's model, device, seed and checkpoint settings before running it.

For the current comparison protocol and reproducible report, see the [benchmark](benchmark/index.md). For individual training stages, see [PA and DPD training](training.md).

### OpenDPDv1
To reproduce the PA modeling results shown in **OpenDPD** Figure 4(a):
```bash
bash bash_scripts/train_all_pa.sh
```
This script trains multiple PA models (each with approximately 500 parameters) using 5 different random seeds. Figure 4(a) displays the averaged results from these runs.

To reproduce the DPD learning results in Figure 4(b), Figure 4(d), and Table 1:
```bash
bash bash_scripts/train_all_dpd.sh
```
This script trains various DPD models, each with approximately 500 parameters.

### Mixed-Precision DPD (MP-DPD)

For convenience, you can reproduce all MP-DPD results using:
```bash
bash bash_scripts/quant_mp_dpd.sh
```

### OpenDPDv2
To reproduce the quantized (W16A16) TRes-DeltaGRU-450 DPD modeling results shown in **OpenDPDv2** Table 1:
```bash
bash bash_scripts/OpenDPDv2.sh
```
