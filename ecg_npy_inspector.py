#!/usr/bin/env python3
"""
ecg_npy_inspector.py
====================
檢查並可視化 generate_ecg_powerline_drift.py 生成的 ECG .npy 文件。

用法:
    python ecg_npy_inspector.py --npy path/to/ECG_256Hz_2s_filtered_0.5-40.0Hz.npy
    python ecg_npy_inspector.py --dir ./artifacts          # 自動查找 ECG*.npy
"""

import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 無頭環境兼容
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def find_ecg_npy(directory):
    """在目錄中查找 ECG*.npy 文件。"""
    patterns = [
        os.path.join(directory, 'ECG*.npy'),
        os.path.join(directory, '*ECG*.npy'),
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            exact = [f for f in files if os.path.basename(f).startswith('ECG_')]
            return exact[0] if exact else files[0]
    return None


def simple_r_peaks(signal_1d, fs, min_distance_ms=300):
    """
    極簡 R-peak 檢測（基於幅度閾值 + 最小間距）。
    僅用於快速驗證，非醫療級檢測。
    """
    signal_hp = signal_1d - np.convolve(signal_1d, np.ones(fs // 4) / (fs // 4), mode='same')
    threshold = np.std(signal_hp) * 2.0
    min_samples = int(min_distance_ms / 1000.0 * fs)

    peaks = []
    for i in range(1, len(signal_hp) - 1):
        if signal_hp[i] > threshold and signal_hp[i] > signal_hp[i - 1] and signal_hp[i] > signal_hp[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_samples:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def inspect_ecg_npy(npy_path, fs=256, n_show=6, save_fig=None):
    """
    加載 ECG npy 並進行全面檢查與可視化。
    """
    print("=" * 60)
    print("ECG .npy 檢查報告")
    print("=" * 60)
    print(f"文件路徑: {npy_path}")
    print(f"文件大小: {os.path.getsize(npy_path) / 1024 / 1024:.2f} MB")

    # 1. 加載數據
    data = np.load(npy_path)
    print(f"[1] 數據形狀: {data.shape}")
    print(f"    數據類型: {data.dtype}")

    if data.ndim != 2:
        print(f"    ⚠️  警告: 期望 2D 數組 (n_epochs, n_samples)，實際為 {data.ndim}D")
        return

    n_epochs, n_samples = data.shape
    duration_sec = n_samples / fs
    print(f"    Epoch 數量: {n_epochs}")
    print(f"    每個 Epoch 採樣點: {n_samples}")
    print(f"    每個 Epoch 時長: {duration_sec:.3f} s")

    # 2. 基本統計檢查
    print("[2] 基本統計檢查")
    print(f"    NaN 數量: {np.isnan(data).sum()}")
    print(f"    Inf 數量: {np.isinf(data).sum()}")
    print(f"    全局均值: {np.mean(data):.4f}  (Z-score 後應接近 0)")
    print(f"    全局標準差: {np.std(data):.4f}  (Z-score 後應接近 1)")
    print(f"    最小值: {np.min(data):.4f}")
    print(f"    最大值: {np.max(data):.4f}")
    print(f"    峰峰值 (P-P): {np.ptp(data):.4f}")

    epoch_means = np.mean(data, axis=1)
    epoch_stds = np.std(data, axis=1)
    print(f"    各 epoch 均值範圍: [{epoch_means.min():.4f}, {epoch_means.max():.4f}]")
    print(f"    各 epoch 標準差範圍: [{epoch_stds.min():.4f}, {epoch_stds.max():.4f}]")

    bad_epochs = []
    for i in range(n_epochs):
        if np.isnan(data[i]).any() or np.isinf(data[i]).any():
            bad_epochs.append(i)
    if bad_epochs:
        print(f"    ⚠️  異常 epoch 索引: {bad_epochs}")
    else:
        print(f"    ✅ 所有 epoch 無 NaN / Inf")

    # 3. 時域波形可視化
    print(f"[3] 繪製隨機 {n_show} 個 epoch 的時域波形...")
    rng = np.random.default_rng(42)
    show_idx = rng.choice(n_epochs, size=min(n_show, n_epochs), replace=False)
    show_idx = sorted(show_idx)

    t = np.arange(n_samples) / fs

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])

    for k, idx in enumerate(show_idx):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.plot(t, data[idx], color='#2E86AB', linewidth=0.8)
        ax.set_title(f'Epoch #{idx}  (mean={epoch_means[idx]:.3f}, std={epoch_stds[idx]:.3f})', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlim([0, duration_sec])
        ax.grid(True, alpha=0.3)
        peaks = simple_r_peaks(data[idx], fs)
        if len(peaks) > 0:
            ax.scatter(t[peaks], data[idx][peaks], color='red', s=20, zorder=5, label=f'R-peaks ≈{len(peaks)}')
            ax.legend(fontsize=8, loc='upper right')

    # 疊加均值波形
    ax_mean = fig.add_subplot(gs[2, :])
    mean_wave = np.mean(data[show_idx], axis=0)
    std_wave = np.std(data[show_idx], axis=0)
    ax_mean.plot(t, mean_wave, color='#A23B72', linewidth=1.5, label=f'Mean of {len(show_idx)} epochs')
    ax_mean.fill_between(t, mean_wave - std_wave, mean_wave + std_wave, color='#A23B72', alpha=0.2, label='±1 std')
    ax_mean.set_title('Overlay: Mean ± Std of displayed epochs', fontsize=11)
    ax_mean.set_xlabel('Time (s)')
    ax_mean.set_ylabel('Amplitude (a.u.)')
    ax_mean.set_xlim([0, duration_sec])
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc='upper right')

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"    時域圖已保存: {save_fig}")
    plt.close(fig)

    # 4. 頻譜檢查
    print("[4] 頻譜檢查 (平均 FFT)...")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    fft_vals = np.fft.rfft(data[show_idx[0]])
    freqs = np.fft.rfftfreq(n_samples, 1 / fs)
    axes[0].plot(freqs, 20 * np.log10(np.abs(fft_vals) + 1e-12), color='#2E86AB', linewidth=0.8)
    axes[0].set_title(f'FFT of Epoch #{show_idx[0]}')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_xlim([0, fs // 2])
    axes[0].axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz notch')
    axes[0].axvline(40, color='green', linestyle='--', alpha=0.5, label='40 Hz cutoff')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    all_fft = np.fft.rfft(data, axis=1)
    mean_fft = np.mean(np.abs(all_fft), axis=0)
    axes[1].plot(freqs, 20 * np.log10(mean_fft + 1e-12), color='#F18F01', linewidth=1.0)
    axes[1].set_title(f'Average FFT of all {n_epochs} epochs')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_xlim([0, fs // 2])
    axes[1].axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz notch')
    axes[1].axvline(40, color='green', linestyle='--', alpha=0.5, label='40 Hz cutoff')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        spec_path = save_fig.replace('.png', '_spectrum.png')
        plt.savefig(spec_path, dpi=150, bbox_inches='tight')
        print(f"    頻譜圖已保存: {spec_path}")
    plt.close(fig2)

    # 5. 心率估計
    print("[5] 心率估計檢查 (簡易 R-peak 檢測)...")
    hr_list = []
    for i in range(min(50, n_epochs)):
        peaks = simple_r_peaks(data[i], fs)
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fs
            hr = 60.0 / np.mean(rr_intervals)
            hr_list.append(hr)

    if hr_list:
        hr_arr = np.array(hr_list)
        print(f"    檢測到 {len(hr_list)} 個有效 epoch 的心率")
        print(f"    心率範圍: [{hr_arr.min():.1f}, {hr_arr.max():.1f}] BPM")
        print(f"    心率均值: {hr_arr.mean():.1f} BPM  (正常成人 60-100 BPM)")
        if hr_arr.mean() < 50 or hr_arr.mean() > 120:
            print(f"    ⚠️  平均心率異常，請檢查數據或 R-peak 檢測閾值")
        else:
            print(f"    ✅ 心率在合理範圍內")
    else:
        print(f"    ⚠️  未能檢測到足夠 R-peaks，數據可能過短或幅度過小")

    # 6. 總結
    print("" + "=" * 60)
    print("檢查總結")
    print("=" * 60)
    checks = [
        ("形狀正確 (n_epochs, n_samples)", data.ndim == 2),
        ("無 NaN", np.isnan(data).sum() == 0),
        ("無 Inf", np.isinf(data).sum() == 0),
        ("Z-score 均值接近 0", abs(np.mean(data)) < 0.1),
        ("Z-score 標準差接近 1", 0.8 < np.std(data) < 1.2),
        ("心率合理 (若檢測到)", len(hr_list) == 0 or (50 <= np.mean(hr_arr) <= 120)),
    ]
    for desc, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {desc}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='檢查 ECG .npy 文件')
    parser.add_argument('--npy', type=str, default=None, help='ECG .npy 文件路徑')
    parser.add_argument('--dir', type=str, default='./artifacts', help='搜索目錄（自動查找 ECG*.npy）')
    parser.add_argument('--fs', type=int, default=256, help='採樣率 (Hz)')
    parser.add_argument('--n-show', type=int, default=6, help='顯示的 epoch 數量')
    parser.add_argument('--save', type=str, default='ecg_check.png', help='保存圖片路徑（如 ecg_check.png）')
    args = parser.parse_args()

    npy_path = args.npy
    if npy_path is None:
        npy_path = find_ecg_npy(args.dir)
        if npy_path is None:
            print(f"錯誤: 在 {args.dir} 中未找到 ECG*.npy 文件")
            print("請使用 --npy 指定文件路徑，或確認 --dir 目錄正確")
            sys.exit(1)
        print(f"自動找到文件: {npy_path}")

    if not os.path.exists(npy_path):
        print(f"錯誤: 文件不存在: {npy_path}")
        sys.exit(1)

    inspect_ecg_npy(npy_path, fs=args.fs, n_show=args.n_show, save_fig=args.save)


if __name__ == '__main__':
    main()
