import numpy as np


def NMSE(prediction, ground_truth):
    i_hat = prediction[..., 0]
    i_true = ground_truth[..., 0]
    q_hat = prediction[..., 1]
    q_true = ground_truth[..., 1]

    MSE = np.mean(np.square(i_true - i_hat) + np.square(q_true - q_hat), axis=-1)
    energy = np.mean(np.square(i_true) + np.square(q_true), axis=-1)

    NMSE = np.mean(10 * np.log10(MSE / energy))
    return NMSE


def EVM(prediction, desired_outputs, nperseg=None):
    """
    Calculate EVM based on the given formula.

    Parameters:
    - prediction: Prediction/measurements of the PA or DPD-PA outputs.
    - desired_outputs: Desired Output of PA
    - nperseg: Length of each segment for the FFT. If None, use the entire signal.

    Returns:
    - EVM value in dB.
    """

    # If nperseg is not provided, set it to the length of the signal
    if nperseg is None:
        nperseg = len(prediction)

    i_hat = prediction[..., 0]
    i_true = desired_outputs[..., 0]
    q_hat = prediction[..., 1]
    q_true = desired_outputs[..., 1]

    # Compute FFT for reference signal
    i_true_fft = np.fft.fft(i_true, n=nperseg)
    q_true_fft = np.fft.fft(q_true, n=nperseg)

    # Compute FFT for measured signal
    i_hat_fft = np.fft.fft(i_hat, n=nperseg)
    q_hat_fft = np.fft.fft(q_hat, n=nperseg)

    numerator = np.sum((i_hat_fft - i_true_fft).real ** 2 + (q_hat_fft - q_true_fft).real ** 2)
    denominator = np.sum(i_true_fft.real ** 2 + q_true_fft.real ** 2)

    evm = np.sqrt(numerator / denominator)

    # Convert to dB
    evm_db = 20 * np.log10(evm)

    return evm_db


def ACLR(IQ_signal, fs=800e6, nperseg=2560, bw_main_ch=200e6, bw_side_ch=20e6):
    """
    Calculates the left and right Adjacent Channel Leakage Ratio (ACLR).

    Parameters:
    - frequencies: Array of frequency values from the Welch method.
    - psd: Power Spectral Density values corresponding to the frequencies.
    - bw_main_ch: Bandwidth of the main channel baseband signals (default is 200e6).
    - bw_side_ch: Bandwidth of the side channel baseband signals (default is 20e6).

    Returns:
    - aclr_left: ACLR value for the left adjacent channel.
    - aclr_right: ACLR value for the right adjacent channel.
    """
    # Get complex signal
    complex_signal = IQ_to_complex(IQ_signal)

    # Calculate power spectral density
    freq, psd = PSD(complex_signal, fs=fs, nperseg=nperseg)

    # Calculate the indices for the main and adjacent channels
    main_channel_indices = np.where((freq >= -bw_main_ch / 2) & (freq < bw_main_ch / 2))[0]
    main_left_channel_indices = np.where((freq >= -bw_main_ch / 2) & (freq < -bw_main_ch / 2 + bw_side_ch))[0]
    main_right_channel_indices = np.where((freq >= bw_main_ch / 2 - bw_side_ch) & (freq < bw_main_ch / 2))[0]
    left_adj_channel_indices = np.where((freq >= -bw_main_ch / 2 - bw_side_ch) & (freq < -bw_main_ch / 2))[0]
    right_adj_channel_indices = np.where((freq >= bw_main_ch / 2) & (freq < bw_main_ch / 2 + bw_side_ch))[0]

    # Compute the power in the main and adjacent channels
    power_main_channel = np.sum(psd[main_channel_indices])
    power_left_main_channel = np.sum(psd[main_left_channel_indices])
    power_right_main_channel = np.sum(psd[main_right_channel_indices])
    power_left_adj_channel = np.sum(psd[left_adj_channel_indices])
    power_right_adj_channel = np.sum(psd[right_adj_channel_indices])

    # Compute ACLR for left and right adjacent channels
    aclr_left = 10 * np.log10(power_left_adj_channel / power_left_main_channel)
    aclr_right = 10 * np.log10(power_right_adj_channel / power_right_main_channel)

    return aclr_left, aclr_right


def PSD(complex_signal, fs=800e6, nperseg=2560):
    """
    Compute the Power Spectral Density (PSD) of a given complex signal using the Welch method.

    Parameters:
    - complex_signal: Input complex signal for which the PSD is to be computed.
    - fs (float, optional): Sampling frequency of the signal. Default is 800e6 (800 MHz).
    - nperseg (int, optional): Number of data points to be used in each block for the Welch method. Default is 2560.

    Returns:
    - frequencies_signal_subset: Frequencies at which the PSD is computed.
    - psd_signal_subset: PSD values.
    """

    import numpy as np
    from scipy.signal import welch

    # Use only the specified number of data points from the beginning of the input signal
    complex_signal_subset = complex_signal[:nperseg]

    # Compute the PSD using the Welch method
    freq, psd = welch(complex_signal_subset, fs=fs, nperseg=nperseg,
                      return_onesided=False)

    # To make the frequency axis monotonic, we need to shift the zero frequency component to the center.
    # This step rearranges the computed PSD and frequency values such that the negative frequencies appear first.
    half_nfft = int(nperseg / 2)
    freq = np.concatenate(
        (freq[half_nfft:], freq[:half_nfft]))

    # Rearrange the PSD values corresponding to the rearranged frequency values.
    psd = np.concatenate((psd[half_nfft:], psd[:half_nfft]))

    return freq, psd


def IQ_to_complex(IQ_signal):
    """
    Convert a multi-dimensional array of I-Q pairs into a 2D array of complex signals.

    Args:
    - IQ_in_segment (3D array): The prediction I-Q data with shape (#segments, frame_length, 2).

    Returns:
    - 2D array of shape (#segments, frame_length) containing complex signals.
    """

    # Extract I and Q values
    I_values = IQ_signal[..., 0]
    Q_values = IQ_signal[..., 1]

    # Convert to complex signals
    complex_signals = I_values + 1j * Q_values

    return complex_signals