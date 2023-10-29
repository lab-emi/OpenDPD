import numpy as np
import graycode
from utils import filters,fft_wrappers




class OFDM:
    def __init__(self,fftlength, guardband, cpprefix, cpprefixlength, pilotnum, pilot_index,
                 pilotValue, channelnum, channel_BW, inputrms, m_QAM, oversampling_rate, beta):
        self.fftlength = fftlength
        self.guardband = guardband
        self.cpprefix = cpprefix
        self.cpprefixlength = cpprefixlength
        self.pilotnum = pilotnum
        self.pilot_index = pilot_index
        self.pilotValue = pilotValue
        self.channelnum = int(channelnum/2)
        self.channel_BW = channel_BW
        self.Pin = inputrms
        self.m_QAM = m_QAM
        self.osr = oversampling_rate
        self.beta = 0.2
        self.window_length = oversampling_rate*fftlength*channelnum
        self.Ts = 1/channel_BW
        self.Fs = channel_BW*channelnum*oversampling_rate


    def OFDM_signal_generate(self):
        OFDM_time = np.zeros(self.fftlength*self.osr, dtype=complex)

        # generate m-QAM mapping table
        axis_length = int(np.sqrt(self.m_QAM))
        x_axis = {}
        y_axis = {}
        mapping_table = {}
        for i in range(1, 6):
            if pow(2, i) == axis_length:
                mu = i
        form = '{:0' + str(mu) + 'b}'
        for i in range(axis_length):
            x_axis[-axis_length + 1 + i * 2] = form.format(graycode.tc_to_gray_code(i))
            y_axis[-axis_length + 1 + i * 2] = form.format(graycode.tc_to_gray_code(i))
        for m in x_axis.keys():
            for n in y_axis.keys():
                key = x_axis[m] + y_axis[n]
                value = np.complex(m, n)
                mapping_table[key] = value

        for channel_index in range(-self.channelnum,self.channelnum+1):
            K = self.fftlength
            allCarriers = np.arange(K)
            dataCarriers = allCarriers


            # assign pilot value to subcarriers with pilot index
            if self.pilotnum != 0:
                if self.pilot_index == None:
                    pilotCarriers = allCarriers[::K // self.pilotnum]
                    # delete pilot carriers in all carriers
                    dataCarriers = np.delete(allCarriers, pilotCarriers)
                else:
                    pilotCarriers = allCarriers[self.pilot_index]
                    # delete pilot carriers in all carriers
                    dataCarriers = np.delete(allCarriers, pilotCarriers)
                OFDM_data[pilotCarriers] = self.pilotValue  # allocate the pilot subcarriers

            #delete guard band carriers in all carriers
            guardCarriers1 = allCarriers[0:self.guardband+1]
            dataCarriers = np.delete(dataCarriers, guardCarriers1)
            guardCarriers2 = allCarriers[K-self.guardband:K]
            dataCarriers = np.delete(dataCarriers, guardCarriers2)

            #all bits that are needed for one channel
            payloadBits_per_OFDM = len(dataCarriers) * 2*mu

            #generate fftlength-2*guardband-1-pilotnumber sets binary code and map them to QAM
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            bits_SP = bits.reshape((len(dataCarriers), 2*mu))
            QAM = np.array([mapping_table[str(b)[1:-1].replace(' ','')] for b in bits_SP])
            OFDM_data = np.zeros(K, dtype=complex)  # the overall K subcarriers
            OFDM_data[dataCarriers] = QAM
            # OFDM_f = np.roll(OFDM_data, int(len(OFDM_data)/2))
            OFDM_f = OFDM_data
            #frequency shift and upsampling in baseband
            OFDM_t = np.fft.ifft(OFDM_f)
            up_rz = np.zeros(K * self.osr, dtype=OFDM_time.dtype)
            up_rz[::self.osr] = OFDM_t
            up_rz_f = np.zeros(K * self.osr, dtype=complex)
            up_rz_freq = np.fft.fft(up_rz)
            up_rz_f[:self.fftlength]=up_rz_freq[:self.fftlength]
            up_rz_f = np.roll(up_rz_f, -(channel_index*K+int(self.fftlength /2)))
            up_rz_t = np.fft.ifft(up_rz_f)

            # multiply with spectral mask in frequency domain
            OFDM_time = np.add(OFDM_time, up_rz_t)


        # add cylic prefix
        if self.cpprefix == True:
            if self.cpprefix <= self.fftlength * self.channelnum / 4:
                CP = self.cpprefixlength
            else:
                raise RuntimeError('Please set prefix length less than 1/4 FFT length.')
            cp = OFDM_time[-CP:]
            OFDM_time = np.hstack([cp, OFDM])


        # scale the signal to stay within the input range of PA
        scaler = self.Pin/np.max(np.abs(OFDM_time))
        OFDM_time = OFDM_time*scaler

        return OFDM_time






