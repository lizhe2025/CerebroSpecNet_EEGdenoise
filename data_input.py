import numpy as np
import math

# Author: Haoming Zhang (Refactored for composite 3-artifact noise: EOG+EMG+ECG)
# All operations vectorized and forced float32.


def get_rms(records):
    records = np.asarray(records, dtype=np.float32)
    return np.sqrt(np.mean(records ** 2))


def random_signal(signal, combin_num):
    """Random disturb and augment. Pre-allocated float32 array."""
    signal = np.asarray(signal, dtype=np.float32)
    n_samples, n_points = signal.shape
    result = np.empty((combin_num * n_samples, n_points), dtype=np.float32)
    for i in range(combin_num):
        perm = np.random.permutation(n_samples)
        result[i * n_samples:(i + 1) * n_samples] = signal[perm]
    return result


def data_prepare(EEG_all, noise_all, combin_num, train_num, test_num):
    """
    SINGLE artifact mixing (legacy, kept for single-artifact test sets).
    """
    EEG_all = np.asarray(EEG_all, dtype=np.float32)
    noise_all = np.asarray(noise_all, dtype=np.float32)
    n_points = EEG_all.shape[1]

    # Train
    eeg_train = EEG_all[:train_num]
    noise_train = noise_all[:train_num]

    EEG_train = random_signal(eeg_train, combin_num)
    NOISE_train = random_signal(noise_train, combin_num)
    n_train_total = EEG_train.shape[0]

    SNR_train_dB = np.random.uniform(-7.0, 2.0, n_train_total).astype(np.float32)
    SNR_train = 10.0 ** (0.1 * SNR_train_dB)

    rms_eeg = np.sqrt(np.mean(EEG_train ** 2, axis=1))
    rms_noise = np.sqrt(np.mean(NOISE_train ** 2, axis=1))
    coe = rms_eeg / (rms_noise * SNR_train + 1e-12)

    NOISE_train_adjust = NOISE_train * coe[:, None]
    noiseEEG_train = EEG_train + NOISE_train_adjust

    std_vals = np.std(noiseEEG_train, axis=1, keepdims=True) + 1e-12
    EEG_train_end_standard = EEG_train / std_vals
    noiseEEG_train_end_standard = noiseEEG_train / std_vals

    # Test
    eeg_test = EEG_all[train_num:train_num + test_num]
    noise_test = noise_all[train_num:train_num + test_num]

    SNR_test_dB = np.linspace(-7.0, 2.0, num=10).astype(np.float32)
    SNR_test = 10.0 ** (0.1 * SNR_test_dB)

    rms_eeg_t = np.sqrt(np.mean(eeg_test ** 2, axis=1))
    rms_noise_t = np.sqrt(np.mean(noise_test ** 2, axis=1))
    coe_t = rms_eeg_t[None, :] / (rms_noise_t[None, :] * SNR_test[:, None] + 1e-12)

    noise_eeg_test = noise_test[None, :, :] * coe_t[:, :, None]
    noiseEEG_test = eeg_test[None, :, :] + noise_eeg_test
    noiseEEG_test = noiseEEG_test.reshape(-1, n_points)
    EEG_test_tiled = np.tile(eeg_test, (10, 1))

    std_test = np.std(noiseEEG_test, axis=1, keepdims=True) + 1e-12
    EEG_test_end_standard = EEG_test_tiled / std_test
    noiseEEG_test_end_standard = noiseEEG_test / std_test
    std_VALUE = std_test.squeeze().astype(np.float32)

    return (
        noiseEEG_train_end_standard.astype(np.float32),
        EEG_train_end_standard.astype(np.float32),
        noiseEEG_test_end_standard.astype(np.float32),
        EEG_test_end_standard.astype(np.float32),
        std_VALUE,
    )


def data_prepare_multi(EEG_all, noise_list, combin_num, train_num, test_num,
                       snr_ranges=None):
    """
    COMPOSITE noise: EACH EEG sample is mixed with ALL artifact types simultaneously.
    (3-artifact version: EOG + EMG + ECG)

    Parameters
    ----------
    EEG_all : (N, T) clean EEG
    noise_list : list of 3 arrays [EOG, EMG, ECG], each (M, T)
    combin_num : int
    train_num, test_num : int
    snr_ranges : list of 3 tuples, e.g. [(-7,2), (-7,2), (-7,2)]
                 SNR in dB for each noise type independently.

    Returns
    -------
    Same 5-tuple as data_prepare, but noise is composite (sum of 3 scaled artifacts).
    """
    EEG_all = np.asarray(EEG_all, dtype=np.float32)
    noise_list = [np.asarray(n, dtype=np.float32) for n in noise_list]
    n_points = EEG_all.shape[1]
    n_types = len(noise_list)

    if snr_ranges is None:
        snr_ranges = [(-7.0, 2.0), (-7.0, 2.0), (-7.0, 2.0)]

    required = train_num + test_num
    for i, noise in enumerate(noise_list):
        if noise.shape[0] < required:
            idx = np.random.choice(noise.shape[0], required, replace=True)
            noise_list[i] = noise[idx]
        else:
            noise_list[i] = noise[:required]

    # ==================== TRAIN: composite noise ====================
    eeg_train = EEG_all[:train_num]
    EEG_train = random_signal(eeg_train, combin_num)
    noise_trains = [random_signal(n[:train_num], combin_num) for n in noise_list]
    n_train_total = EEG_train.shape[0]

    composite_noise = np.zeros_like(EEG_train)
    for noise_train, (snr_low, snr_high) in zip(noise_trains, snr_ranges):
        SNR_train_dB = np.random.uniform(snr_low, snr_high, n_train_total).astype(np.float32)
        SNR_train = 10.0 ** (0.1 * SNR_train_dB)

        rms_eeg = np.sqrt(np.mean(EEG_train ** 2, axis=1))
        rms_noise = np.sqrt(np.mean(noise_train ** 2, axis=1))
        coe = rms_eeg / (rms_noise * SNR_train + 1e-12)

        composite_noise += noise_train * coe[:, None]

    noiseEEG_train = EEG_train + composite_noise

    std_vals = np.std(noiseEEG_train, axis=1, keepdims=True) + 1e-12
    EEG_train_end_standard = EEG_train / std_vals
    noiseEEG_train_end_standard = noiseEEG_train / std_vals

    # ==================== TEST: composite noise ====================
    if test_num > 0:
        eeg_test = EEG_all[train_num:train_num + test_num]
        noise_tests = [n[train_num:train_num + test_num] for n in noise_list]

        SNR_test_dB = np.linspace(-7.0, 2.0, num=10).astype(np.float32)
        SNR_test = 10.0 ** (0.1 * SNR_test_dB)

        noise_test_composite = np.zeros((10, test_num, n_points), dtype=np.float32)
        individual_snr = SNR_test * np.sqrt(n_types)

        for noise_test in noise_tests:
            rms_eeg_t = np.sqrt(np.mean(eeg_test ** 2, axis=1))
            rms_noise_t = np.sqrt(np.mean(noise_test ** 2, axis=1))
            coe_t = rms_eeg_t[None, :] / (rms_noise_t[None, :] * individual_snr[:, None] + 1e-12)
            noise_test_composite += noise_test[None, :, :] * coe_t[:, :, None]

        noiseEEG_test = eeg_test[None, :, :] + noise_test_composite
        noiseEEG_test = noiseEEG_test.reshape(-1, n_points)
        EEG_test_tiled = np.tile(eeg_test, (10, 1))

        std_test = np.std(noiseEEG_test, axis=1, keepdims=True) + 1e-12
        EEG_test_end_standard = EEG_test_tiled / std_test
        noiseEEG_test_end_standard = noiseEEG_test / std_test
        std_VALUE = std_test.squeeze().astype(np.float32)
    else:
        noiseEEG_test_end_standard = np.empty((0, n_points), dtype=np.float32)
        EEG_test_end_standard = np.empty((0, n_points), dtype=np.float32)
        std_VALUE = np.empty((0,), dtype=np.float32)

    return (
        noiseEEG_train_end_standard.astype(np.float32),
        EEG_train_end_standard.astype(np.float32),
        noiseEEG_test_end_standard.astype(np.float32),
        EEG_test_end_standard.astype(np.float32),
        std_VALUE,
    )