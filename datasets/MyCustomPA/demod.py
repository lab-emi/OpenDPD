"""Synthetic 64QAM — dummy dataset for tutorial purpose.

IFFT-frame demodulation (nperseg=2560, no cyclic prefix).
"""

from datasets.demodulator import IFFTFrameDemodulator


class Demodulator(IFFTFrameDemodulator):
    pass
