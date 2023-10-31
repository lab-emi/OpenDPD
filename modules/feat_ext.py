import numpy as np


def extract_feature(X, backbone):
    i_x = np.expand_dims(X[..., 0], axis=-1)
    q_x = np.expand_dims(X[..., 1], axis=-1)
    amp2 = np.power(i_x, 2) + np.power(q_x, 2)
    amp = np.sqrt(amp2)
    amp3 = np.power(amp, 3)
    angle = np.angle(i_x + 1j * q_x)
    cos = i_x / amp
    sin = q_x / amp
    if backbone == 'lstm' or backbone == 'gru' or backbone == 'fc':
        features = X
    elif backbone == 'pgjanet' or backbone == 'dvrjanet':
        features = np.concatenate((amp, angle), axis=-1)
    elif backbone == 'rvtdcnn':
        features = np.concatenate((i_x, q_x, amp, amp2, amp3), axis=-1)
    elif backbone == 'dgru':
        features = np.concatenate((i_x, q_x, amp, amp3, sin, cos), axis=-1)
    else:
        features = X
    return features