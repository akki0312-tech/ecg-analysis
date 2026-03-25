"""
ECG Comprehensive Comparison Analysis
======================================
Analyzes 3 ECG records using 5 signal processing methods and
diagnoses each using threshold-based voting (no ML needed).

Records:
  100 - Normal Sinus Rhythm
  106 - Ventricular Arrhythmia
  207 - Atrial Fibrillation
"""

import wfdb
import numpy as np
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. CORE SIGNAL PROCESSING FUNCTIONS
# ──────────────────────────────────────────────

def load_ecg(record_name):
    record = wfdb.rdrecord(record_name, pn_dir='mitdb')
    ecg = record.p_signal[:10000, 0]
    fs = record.fs  # 360 Hz
    return ecg, fs

def preprocess_ecg(ecg, fs):
    sos = signal.butter(4, 0.5, btype='high', fs=fs, output='sos')
    ecg_clean = signal.sosfiltfilt(sos, ecg)
    b, a = signal.iirnotch(60, 30, fs)
    ecg_clean = signal.filtfilt(b, a, ecg_clean)
    return ecg_clean

def detect_r_peaks(ecg, fs):
    diff_ecg = np.diff(ecg)
    squared = diff_ecg ** 2
    window = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(integrated, distance=int(0.2*fs), height=np.mean(integrated))
    return peaks

# ──────────────────────────────────────────────
# 2. ANALYSIS METHODS (return params + diagnosis)
# ──────────────────────────────────────────────

def histogram_analysis(ecg, fs):
    r_peaks = detect_r_peaks(ecg, fs)
    rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
    p = {
        'mean_rr':  np.mean(rr_intervals),
        'std_rr':   np.std(rr_intervals),
        'cv_rr':    np.std(rr_intervals) / np.mean(rr_intervals),
        'min_rr':   np.min(rr_intervals),
        'max_rr':   np.max(rr_intervals),
        'range_rr': np.max(rr_intervals) - np.min(rr_intervals),
        'mean_hr':  60000 / np.mean(rr_intervals),
    }
    flags = {
        'std_rr>100':    p['std_rr'] > 100,
        'CV>0.15':       p['cv_rr'] > 0.15,
        'range>400ms':   p['range_rr'] > 400,
    }
    abnormal = any(flags.values())
    return rr_intervals, p, flags, abnormal

def wavelet_analysis(ecg, fs):
    coeffs = pywt.wavedec(ecg, 'db4', level=5)
    energies = [np.sum(c**2) for c in coeffs]
    total = sum(energies)
    ep = [e/total*100 for e in energies]
    p = {
        'energy_distribution': ep,
        'detail_energy':       sum(ep[1:]),
        'approx_energy':       ep[0],
        'dominant_level':      np.argmax(ep),
    }
    flags = {
        'detail_energy>60%': p['detail_energy'] > 60,
        'dominant>level2':   p['dominant_level'] > 2,
    }
    abnormal = any(flags.values())
    return coeffs, p, flags, abnormal

def stft_analysis(ecg, fs):
    f, t, Zxx = signal.stft(ecg, fs, nperseg=256, noverlap=250)
    spectrogram = np.abs(Zxx) ** 2
    freq_mask = f <= 10
    f_cardiac = f[freq_mask]
    spec_cardiac = spectrogram[freq_mask, :]
    dominant_freqs = f_cardiac[np.argmax(spec_cardiac, axis=0)]
    mean_df = np.mean(dominant_freqs)
    p = {
        'mean_dominant_freq':  mean_df,
        'std_dominant_freq':   np.std(dominant_freqs),
        'freq_stability_cv':   np.std(dominant_freqs) / mean_df if mean_df > 0 else 0,
        'freq_range':          np.max(dominant_freqs) - np.min(dominant_freqs),
    }
    flags = {
        'std_freq>0.3Hz': p['std_dominant_freq'] > 0.3,
        'CV>0.2':         p['freq_stability_cv'] > 0.2,
        'range>1.5Hz':    p['freq_range'] > 1.5,
    }
    abnormal = any(flags.values())
    return f, t, Zxx, p, flags, abnormal

def fft_analysis(ecg, fs):
    N = len(ecg)
    fft_vals = np.fft.fft(ecg)
    fft_freq = np.fft.fftfreq(N, 1/fs)
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    cardiac_mask = (freqs >= 0.5) & (freqs <= 5)
    cardiac_freqs = freqs[cardiac_mask]
    cardiac_power = power[cardiac_mask]
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(cardiac_power, height=np.max(cardiac_power)*0.1)
    peak_freqs = cardiac_freqs[peaks_idx]
    dom_freq = cardiac_freqs[np.argmax(cardiac_power)]
    p = {
        'num_peaks':      len(peak_freqs),
        'dominant_freq':  dom_freq,
        'dominant_hr':    dom_freq * 60,
        'peak_frequencies': peak_freqs,
        'spectral_energy':  np.sum(cardiac_power),
    }
    flags = {
        'peaks>3':        p['num_peaks'] > 3,
        'HR<50 or >120':  p['dominant_hr'] < 50 or p['dominant_hr'] > 120,
        'no_clear_peak':  p['num_peaks'] == 0,
    }
    abnormal = any(flags.values())
    return freqs, power, p, flags, abnormal

def entropy_analysis(ecg, fs):
    ecg_sub = ecg[:2000]  # small subset for speed

    # Shannon
    hist, _ = np.histogram(ecg_sub, bins=50, density=True)
    hist = hist[hist > 0]
    shannon = -np.sum(hist * np.log2(hist))

    # Approximate Entropy (fast)
    def apen(sig, m=2, r=None):
        if r is None: r = 0.2 * np.std(sig)
        N = len(sig)
        def phi(m):
            pats = np.array([sig[i:i+m] for i in range(N-m+1)])
            C = [np.sum(np.max(np.abs(pats - pats[i]), axis=1) <= r) / (N-m+1.0)
                 for i in range(len(pats))]
            return np.mean(np.log(C))
        return abs(phi(m+1) - phi(m))

    # Sample Entropy (fast)
    def sampen(sig, m=2, r=None):
        if r is None: r = 0.2 * np.std(sig)
        N = len(sig)
        def phi(m):
            pats = np.array([sig[i:i+m] for i in range(N-m)])
            B = 0
            for i in range(len(pats)):
                dists = np.max(np.abs(pats - pats[i]), axis=1)
                B += np.sum((dists <= r) & (np.arange(len(pats)) != i))
            return B
        A, B = phi(m+1), phi(m)
        return -np.log(A/B) if B != 0 and A != 0 else 0

    try:
        ap = apen(ecg_sub)
        sp = sampen(ecg_sub)
    except Exception:
        ap, sp = 0, 0

    p = {'shannon_entropy': shannon, 'approximate_entropy': ap, 'sample_entropy': sp}
    flags = {
        'ApEn>1.5':   ap > 1.5,
        'SampEn>1.2': sp > 1.2,
    }
    abnormal = any(flags.values())
    return p, flags, abnormal


# ──────────────────────────────────────────────
# 3. MAIN: RUN ALL RECORDS
# ──────────────────────────────────────────────

CASES = [
    ('100', 'Normal Sinus Rhythm'),
    ('106', 'Ventricular Arrhythmia'),
    ('207', 'Atrial Fibrillation'),
]

print("="*70)
print("  ECG COMPREHENSIVE COMPARISON ANALYSIS")
print("  Threshold-Based Diagnosis (No ML)")
print("="*70)

all_results = {}

for record_id, label in CASES:
    print(f"\n▶ Loading & analyzing Record {record_id} ({label})...")
    ecg_raw, fs = load_ecg(record_id)
    ecg = preprocess_ecg(ecg_raw, fs)

    rr_intervals, h_p, h_f, h_abn    = histogram_analysis(ecg, fs)
    _, w_p, w_f, w_abn               = wavelet_analysis(ecg, fs)
    f_arr, t_arr, Zxx, s_p, s_f, s_abn = stft_analysis(ecg, fs)
    freqs, power, ff_p, ff_f, ff_abn = fft_analysis(ecg, fs)
    e_p, e_f, e_abn                  = entropy_analysis(ecg, fs)

    abnormal_count = sum([h_abn, w_abn, s_abn, ff_abn, e_abn])
    final_diagnosis = "ARRHYTHMIA DETECTED" if abnormal_count >= 3 else "NORMAL SINUS RHYTHM"

    all_results[record_id] = {
        'label':            label,
        'ecg':              ecg,
        'ecg_raw':          ecg_raw,
        'fs':               fs,
        'rr_intervals':     rr_intervals,
        'Zxx':              Zxx, 'f_stft': f_arr, 't_stft': t_arr,
        'freqs_fft':        freqs, 'power_fft': power,
        'hist':             {'params': h_p, 'flags': h_f, 'abnormal': h_abn},
        'wavelet':          {'params': w_p, 'flags': w_f, 'abnormal': w_abn},
        'stft':             {'params': s_p, 'flags': s_f, 'abnormal': s_abn},
        'fft':              {'params': ff_p, 'flags': ff_f, 'abnormal': ff_abn},
        'entropy':          {'params': e_p, 'flags': e_f, 'abnormal': e_abn},
        'abnormal_count':   abnormal_count,
        'final_diagnosis':  final_diagnosis,
    }
    print(f"   ✓ Done  |  Abnormal methods: {abnormal_count}/5  |  → {final_diagnosis}")

print("\n" + "="*70)
print("  ALL RECORDS PROCESSED SUCCESSFULLY")
print("="*70)


# ──────────────────────────────────────────────
# 4. COMPARISON TABLE (TEXT)
# ──────────────────────────────────────────────

print("\n" + "="*70)
print("  PARAMETER COMPARISON TABLE")
print("="*70)

records = list(all_results.keys())
labels  = [all_results[r]['label'][:14].ljust(14) for r in records]
header  = f"{'Parameter':<30} {'R100':>16} {'R106':>16} {'R207':>16}"
print("\n" + header)
print("-"*80)

def row(name, vals, fmt="{:.3f}"):
    cells = [(fmt.format(v) if v is not None else "N/A").rjust(16) for v in vals]
    print(f"{name:<30}" + "".join(cells))

# Histogram
print("\n── HISTOGRAM (RR Intervals) ──")
row("Mean RR (ms)",   [all_results[r]['hist']['params']['mean_rr']  for r in records])
row("Std RR (ms)",    [all_results[r]['hist']['params']['std_rr']   for r in records])
row("CV (Coeff Var)", [all_results[r]['hist']['params']['cv_rr']    for r in records])
row("Range RR (ms)",  [all_results[r]['hist']['params']['range_rr'] for r in records])
row("Mean HR (bpm)",  [all_results[r]['hist']['params']['mean_hr']  for r in records])

# Wavelet
print("\n── WAVELET (Energy %) ──")
row("Approx Energy (%)", [all_results[r]['wavelet']['params']['approx_energy']  for r in records])
row("Detail Energy (%)", [all_results[r]['wavelet']['params']['detail_energy']  for r in records])
row("Dominant Level",    [all_results[r]['wavelet']['params']['dominant_level'] for r in records], fmt="{:.0f}")

# STFT
print("\n── STFT (Spectral Stability) ──")
row("Mean Dom. Freq (Hz)",  [all_results[r]['stft']['params']['mean_dominant_freq'] for r in records])
row("Std Dom. Freq (Hz)",   [all_results[r]['stft']['params']['std_dominant_freq']  for r in records])
row("Freq Stability CV",    [all_results[r]['stft']['params']['freq_stability_cv']  for r in records])
row("Freq Range (Hz)",      [all_results[r]['stft']['params']['freq_range']         for r in records])

# FFT
print("\n── FFT (Power Spectrum) ──")
row("Dominant HR (bpm)", [all_results[r]['fft']['params']['dominant_hr']   for r in records])
row("Num Peaks",         [all_results[r]['fft']['params']['num_peaks']     for r in records], fmt="{:.0f}")

# Entropy
print("\n── ENTROPY (Complexity) ──")
row("Shannon Entropy",      [all_results[r]['entropy']['params']['shannon_entropy']      for r in records])
row("Approximate Entropy",  [all_results[r]['entropy']['params']['approximate_entropy']  for r in records])
row("Sample Entropy",       [all_results[r]['entropy']['params']['sample_entropy']       for r in records])

print("\n" + "-"*80)

# ──────────────────────────────────────────────
# 5. VOTING SUMMARY TABLE
# ──────────────────────────────────────────────

print("\n" + "="*70)
print("  THRESHOLD-BASED DIAGNOSIS VOTING SUMMARY")
print("="*70)

methods = ['hist', 'wavelet', 'stft', 'fft', 'entropy']
method_names = ['Histogram', 'Wavelet', 'STFT', 'FFT', 'Entropy']

print(f"\n{'Method':<16}", end="")
for r in records:
    lbl = all_results[r]['label'][:14]
    print(f"  {lbl:<18}", end="")
print()
print("-"*80)

for m, mn in zip(methods, method_names):
    print(f"{mn:<16}", end="")
    for r in records:
        abn = all_results[r][m]['abnormal']
        tag = "❌ ABNORMAL" if abn else "✅ NORMAL  "
        print(f"  {tag:<18}", end="")
    print()

print("-"*80)
print(f"{'Abnormal Count':<16}", end="")
for r in records:
    cnt = all_results[r]['abnormal_count']
    print(f"  {str(cnt)+'/5':<18}", end="")
print()

print(f"\n{'FINAL DIAGNOSIS':<16}", end="")
for r in records:
    diag = all_results[r]['final_diagnosis']
    print(f"  {diag[:18]:<18}", end="")
print()
print("="*70)


# ──────────────────────────────────────────────
# 6. VISUALIZATION
# ──────────────────────────────────────────────

COLORS = {'100': '#2ecc71', '106': '#e74c3c', '207': '#e67e22'}
LABELS  = {r: all_results[r]['label'] for r in records}

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor('#1a1a2e')
gs = GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)

def bar_compare(ax, title, values_dict, ylabel, threshold=None, thresh_label=None, color_override=None):
    """Generic 3-bar comparison chart."""
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    rec_ids = list(values_dict.keys())
    vals = list(values_dict.values())
    colors = [COLORS[r] for r in rec_ids] if not color_override else color_override

    bars = ax.bar([LABELS[r][:12] for r in rec_ids], vals, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    if threshold is not None:
        ax.axhline(threshold, color='yellow', linestyle='--', linewidth=1.5,
                   label=f'Threshold: {threshold}' + (f' ({thresh_label})' if thresh_label else ''))
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    ax.grid(axis='y', alpha=0.3, color='#555')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color='white')

# ROW 0: ECG Signals
for i, r in enumerate(records):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#16213e')
    ecg = all_results[r]['ecg']
    fs  = all_results[r]['fs']
    t   = np.arange(len(ecg)) / fs
    ax.plot(t[:1000], ecg[:1000], color=COLORS[r], linewidth=0.8)
    ax.set_title(f"Record {r}\n{LABELS[r]}", fontsize=9, fontweight='bold', color='white')
    ax.set_xlabel('Time (s)', fontsize=7, color='white')
    ax.set_ylabel('Amplitude', fontsize=7, color='white')
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values(): spine.set_color('#444')
    ax.grid(True, alpha=0.2, color='#555')

# ROW 1: Histogram params
ax10 = fig.add_subplot(gs[1, 0])
bar_compare(ax10, "Std RR Interval (ms)\n[Threshold: 100ms]",
            {r: all_results[r]['hist']['params']['std_rr'] for r in records},
            "Std RR (ms)", threshold=100)

ax11 = fig.add_subplot(gs[1, 1])
bar_compare(ax11, "RR Coefficient of Variation\n[Threshold: 0.15]",
            {r: all_results[r]['hist']['params']['cv_rr'] for r in records},
            "CV", threshold=0.15)

ax12 = fig.add_subplot(gs[1, 2])
bar_compare(ax12, "RR Range (ms)\n[Threshold: 400ms]",
            {r: all_results[r]['hist']['params']['range_rr'] for r in records},
            "Range (ms)", threshold=400)

# ROW 2: Wavelet + STFT
ax20 = fig.add_subplot(gs[2, 0])
bar_compare(ax20, "Wavelet Detail Energy (%)\n[Threshold: 60%]",
            {r: all_results[r]['wavelet']['params']['detail_energy'] for r in records},
            "Detail Energy (%)", threshold=60)

ax21 = fig.add_subplot(gs[2, 1])
bar_compare(ax21, "STFT Freq Stability CV\n[Threshold: 0.2]",
            {r: all_results[r]['stft']['params']['freq_stability_cv'] for r in records},
            "CV", threshold=0.2)

ax22 = fig.add_subplot(gs[2, 2])
bar_compare(ax22, "FFT Dominant Heart Rate (bpm)\n[50–120 bpm = Normal]",
            {r: all_results[r]['fft']['params']['dominant_hr'] for r in records},
            "Heart Rate (bpm)")
ax22.set_facecolor('#16213e')
ax22.axhspan(50, 120, alpha=0.12, color='green', label='Normal Zone (50-120)')
ax22.legend(fontsize=7, facecolor='#222', labelcolor='white')

# ROW 3: Entropy
ax30 = fig.add_subplot(gs[3, 0])
bar_compare(ax30, "Approximate Entropy\n[Threshold: > 1.5 = Abnormal]",
            {r: all_results[r]['entropy']['params']['approximate_entropy'] for r in records},
            "ApEn", threshold=1.5)

ax31 = fig.add_subplot(gs[3, 1])
bar_compare(ax31, "Sample Entropy\n[Threshold: > 1.2 = Abnormal]",
            {r: all_results[r]['entropy']['params']['sample_entropy'] for r in records},
            "SampEn", threshold=1.2)

ax32 = fig.add_subplot(gs[3, 2])
bar_compare(ax32, "Shannon Entropy\n[Higher = More Complex]",
            {r: all_results[r]['entropy']['params']['shannon_entropy'] for r in records},
            "Shannon Entropy")

# ROW 4: Voting heatmap + Diagnosis Summary
ax40 = fig.add_subplot(gs[4, :2])
ax40.set_facecolor('#16213e')

vote_matrix = []
for m in methods:
    row_vals = [1 if all_results[r][m]['abnormal'] else 0 for r in records]
    vote_matrix.append(row_vals)

vote_matrix = np.array(vote_matrix)
cmap = plt.cm.RdYlGn_r
im = ax40.imshow(vote_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

ax40.set_xticks(range(len(records)))
ax40.set_xticklabels([f"R{r}\n{LABELS[r][:12]}" for r in records], fontsize=9, color='white')
ax40.set_yticks(range(len(method_names)))
ax40.set_yticklabels(method_names, fontsize=9, color='white')
ax40.set_title("Method Voting Heatmap\n(Green=Normal, Red=Abnormal)", fontsize=10, fontweight='bold', color='white', pad=10)
for spine in ax40.spines.values(): spine.set_color('#444')

for i in range(len(methods)):
    for j in range(len(records)):
        txt = "ABNORMAL" if vote_matrix[i, j] else "NORMAL"
        color = 'white'
        ax40.text(j, i, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=color)

# ROW 4: Final Diagnosis Summary
ax41 = fig.add_subplot(gs[4, 2])
ax41.set_facecolor('#16213e')
ax41.axis('off')

summary_lines = ["FINAL DIAGNOSIS\n"]
for r in records:
    cnt  = all_results[r]['abnormal_count']
    diag = all_results[r]['final_diagnosis']
    icon = "⚠" if "ARRHYTHMIA" in diag else "✓"
    color_tag = "#e74c3c" if "ARRHYTHMIA" in diag else "#2ecc71"
    summary_lines.append(f"{icon}  Record {r}  ({LABELS[r][:14]})")
    summary_lines.append(f"    {cnt}/5 methods abnormal")
    summary_lines.append(f"    → {diag}\n")

ax41.text(0.05, 0.95, "\n".join(summary_lines),
          transform=ax41.transAxes,
          fontsize=8.5, family='monospace',
          verticalalignment='top', color='white',
          bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.7))
ax41.set_title("Diagnosis Summary", fontsize=10, fontweight='bold', color='white', pad=10)

# Overall title
fig.suptitle("ECG Signal Processing Comparison Analysis\n"
             "Records: 100 (Normal) | 106 (Ventricular Arrhythmia) | 207 (Atrial Fibrillation)",
             fontsize=14, fontweight='bold', color='white', y=0.995)

plt.savefig('ecg_comparison_analysis.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("\n✅ Saved: ecg_comparison_analysis.png")
plt.show()
print("\nAnalysis complete!")
