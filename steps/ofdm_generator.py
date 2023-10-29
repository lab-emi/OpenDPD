import pandas as pd
import numpy as np
from modules.OFDMgenerator import OFDM
from utils import fft_wrappers

def main(args):
    signal_size = args.data_size
    ofdm_final = []
    loop = int(signal_size / (args.fftlength * args.oversampling_rate)) + 1

    ofdm = OFDM(fftlength=args.fftlength,
                guardband=args.guardband,
                cpprefix=args.cylic_prefix,
                cpprefixlength=args.cplength,
                pilotnum=args.pilotnum,
                pilot_index=args.pilotindex,
                pilotValue=args.pilot_value,
                channelnum=args.channel_num,
                channel_BW=args.channel_BW,
                inputrms=args.input_rms,
                m_QAM=args.m_QAM,
                oversampling_rate=args.oversampling_rate,
                beta=0)


    for i in range(loop):
        ofdm_time = ofdm.OFDM_signal_generate()
        ofdm_final = np.hstack((ofdm_final, ofdm_time))
    h_rrc_f_freq, h_rrc_f = fft_wrappers.fft_wrapper(ofdm_final, args.channel_BW * args.oversampling_rate, shift=True)
    index_left = np.min(np.where(h_rrc_f_freq >= -args.channel_BW * args.channel_num / 2))
    index_right = np.max(np.where(h_rrc_f_freq <= args.channel_BW * args.channel_num / 2))
    h_f = np.zeros_like(h_rrc_f)
    h_f[index_left:index_right] = h_rrc_f[index_left:index_right]
    h_f = np.roll(h_f, int(len(h_f) / 2))
    ofdm_final = np.fft.ifft(h_f)
    scaler = (args.input_rms) / np.max(np.abs(ofdm_final))
    ofdm_final = ofdm_final * scaler
    df = pd.DataFrame({'I': np.real(ofdm_final[:signal_size]), 'Q': np.imag(ofdm_final[:signal_size])})
    df.to_csv('data/pythonOFDM/Input1.csv', index=False)