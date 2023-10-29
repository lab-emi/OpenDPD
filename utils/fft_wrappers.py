#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: fft.py
# Project: utils
# Created Date: 2023-02-02 17:05:06
# Author: Kuroba
# Description: 
# -----
# Last Modified: 2023-02-02 17:06:50
# Modified By: Kuroba
# -----
# MIT License
# Copyright (c) 2023 Kuroba
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np


def rfft_wrapper(sig, sampling_rate):
    freq = np.fft.rfftfreq(len(sig), d = 1/sampling_rate)
    spectrum = np.fft.rfft(sig)
    return freq, spectrum

def fft_wrapper(input_signal, sampling_rate, shift=False):
    spectrum = np.fft.fft(input_signal)
    if shift:
        spectrum = np.fft.fftshift(spectrum)
        freq = np.fft.fftshift(np.fft.fftfreq(len(input_signal), d = 1/sampling_rate))
    else:
        freq = np.linspace(0, sampling_rate, len(spectrum))
    return freq, spectrum

def fourierSeries(period, N: int) -> np.array:
    # Calculate the Fourier series coefficients up to the N-1th harmonic
    result = []
    T = len(period)
    t = np.arange(T)
    for n in range(N):
        an = 2/T*(period * np.cos(2*np.pi*n*t/T)).sum()
        bn = 2/T*(period * np.sin(2*np.pi*n*t/T)).sum()
        result.append((an, bn))
    return np.array(result)

def reconstruct(P, anbn):
    result = 0
    t = np.arange(P)
    for n, (a, b) in enumerate(anbn):
        if n == 0:
            a = a/2
        result = result + a*np.cos(2*np.pi*n*t/P) + b * np.sin(2*np.pi*n*t/P)
    return result
