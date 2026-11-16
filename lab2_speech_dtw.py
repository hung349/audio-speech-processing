"""CSE457 Lab 2: Vietnamese isolated-word recognition with MFCC + manual DTW.

Expected data layout:
dataset/<label>/*.wav, with labels khong, mot, hai, ba, bon.
Each label needs at least five recordings. First three sorted files train;
remaining files test (matching the Lab 2 case study).

Run: python lab2_speech_dtw.py --data dataset --out results
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# VS Code Code Runner on Windows may use cp1258, which cannot print every
# Vietnamese combining character. Use UTF-8 so diagnostics do not crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FS = 16_000
FRAME_MS, HOP_MS = 25, 10
WIN, HOP = int(FS * FRAME_MS / 1000), int(FS * HOP_MS / 1000)
N_FFT, N_MELS, N_MFCC, PREEMPH = 512, 24, 13, 0.97
LABELS = ["khong", "mot", "hai", "ba", "bon"]
VIETNAMESE = {"khong": "không", "mot": "một", "hai": "hai", "ba": "ba", "bon": "bốn"}


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=FS, mono=True)
    if y.size == 0:
        raise ValueError(f"File rỗng: {path}")
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


def frame_signal(y: np.ndarray) -> np.ndarray:
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    n_frames = 1 + int(np.ceil((len(y) - WIN) / HOP))
    padded = np.pad(y, (0, max(0, (n_frames - 1) * HOP + WIN - len(y))))
    starts = np.arange(n_frames) * HOP
    return np.stack([padded[s:s + WIN] for s in starts])


def time_features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = frame_signal(y)
    windowed = frames * np.hamming(WIN)[None, :]
    energy = np.sum(windowed ** 2, axis=1)
    rms = np.sqrt(np.mean(windowed ** 2, axis=1) + 1e-12)
    signs = np.where(frames >= 0, 1.0, -1.0)
    zcr = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (2 * WIN)
    times = (np.arange(len(frames)) * HOP + WIN / 2) / FS
    return times, energy, rms, zcr


def trim_energy(y: np.ndarray, top_db: float = 35, margin_ms: float = 50,
                zcr_refine: bool = True) -> tuple[np.ndarray, tuple[int, int]]:
    """Relative log-energy endpointing, optionally retain high-ZCR edge frames."""
    times, _, rms, zcr = time_features(y)
    db = 20 * np.log10(rms + 1e-10)
    threshold = max(float(np.max(db)) - top_db, float(np.percentile(db, 15)) + 6.0)
    active = db >= threshold
    if zcr_refine and np.any(active):
        # Unvoiced consonants can have lower energy. Include adjacent frames with
        # appreciable ZCR close to the energetic speech region.
        zcr_edge = zcr >= max(0.08, float(np.percentile(zcr, 70)))
        near = np.convolve(active.astype(int), np.ones(5, dtype=int), mode="same") > 0
        active |= zcr_edge & near
    if not np.any(active):
        return y, (0, len(y))
    first, last = np.flatnonzero(active)[[0, -1]]
    margin = int(FS * margin_ms / 1000)
    start = max(0, int(first * HOP) - margin)
    end = min(len(y), int(last * HOP + WIN) + margin)
    return y[start:end], (start, end)


def mfcc_feature(y: np.ndarray, use_delta: bool = False) -> np.ndarray:
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    emphasized = lfilter([1.0, -PREEMPH], [1.0], y).astype(np.float32)
    feat = librosa.feature.mfcc(
        y=emphasized, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS,
        n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        window="hamming", center=False,
    )
    feat -= np.mean(feat, axis=1, keepdims=True)  # utterance CMN
    result = feat.T
    if use_delta:
        delta = librosa.feature.delta(feat, order=1).T
        result = np.concatenate([result, delta], axis=1)
    return result.astype(np.float32)  # (frames, 13) or (frames, 26)


def dtw_distance(X: np.ndarray, Y: np.ndarray) -> tuple[float, list[tuple[int, int]], np.ndarray]:
    """Manual Euclidean local-distance + 3-neighbor dynamic-programming DTW."""
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        raise ValueError("DTW không nhận chuỗi feature rỗng")
    local = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    back = np.zeros((n + 1, m + 1, 2), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = (acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
            move = int(np.argmin(prev))
            pi, pj = ((i - 1, j), (i, j - 1), (i - 1, j - 1))[move]
            acc[i, j] = local[i - 1, j - 1] + prev[move]
            back[i, j] = (pi, pj)
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = back[i, j]
    path.reverse()
    return float(acc[n, m] / max(1, len(path))), path, local


def plot_waveform_examples(data: Path, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)
    for ax, label in zip(axes, LABELS[:3]):
        files = sorted((data / label).glob("*.wav"))
        if not files:
            continue
        y = load_audio(files[0]); t = np.arange(len(y)) / FS
        ax.plot(t, y, linewidth=.7); ax.set_title(f"{VIETNAMESE[label]} — {files[0].name}")
        ax.set_ylabel("Biên độ"); ax.set_xlabel("Thời gian (s)")
        ax.grid(alpha=.2)
    fig.savefig(out / "waveforms.png", dpi=160); plt.close(fig)


def plot_time_analysis(path: Path, y: np.ndarray, out: Path) -> None:
    t = np.arange(len(y)) / FS
    ft, energy, rms, zcr = time_features(y)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False, constrained_layout=True)
    axes[0].plot(t, y, lw=.7); axes[0].set_ylabel("Biên độ")
    axes[1].plot(ft, 10 * np.log10(energy + 1e-12), label="Energy (dB)")
    axes[1].plot(ft, 20 * np.log10(rms + 1e-12), label="RMS (dB)"); axes[1].legend()
    axes[1].set_ylabel("Mức (dB)")
    axes[2].plot(ft, zcr, color="tab:green"); axes[2].set_ylabel("ZCR"); axes[2].set_xlabel("Thời gian (s)")
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(path.name); fig.savefig(out / f"time_{path.stem}.png", dpi=160); plt.close(fig)


def plot_autocorrelation(path: Path, y: np.ndarray, out: Path) -> None:
    """Illustrate periodicity/pitch on the highest-energy frame."""
    frames = frame_signal(y) * np.hamming(WIN)[None, :]
    idx = int(np.argmax(np.sum(frames ** 2, axis=1)))
    frame = frames[idx] - np.mean(frames[idx])
    corr = np.correlate(frame, frame, mode="full")[WIN - 1:]
    corr /= corr[0] + 1e-12
    # Search plausible adult speech pitch periods (roughly 60--400 Hz).
    lo, hi = max(1, int(FS / 400)), min(len(corr) - 1, int(FS / 60))
    lag = lo + int(np.argmax(corr[lo:hi + 1]))
    pitch = FS / lag
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    lags = np.arange(hi + 1)
    ax.plot(lags, corr[:hi + 1]); ax.axvline(lag, color="tab:red", ls="--", label=f"lag={lag}, F0≈{pitch:.1f} Hz")
    ax.set(title=f"Autocorrelation — {path.name}", xlabel="Lag (mẫu)", ylabel="Chuẩn hóa")
    ax.grid(alpha=.2); ax.legend(); fig.savefig(out / f"autocorrelation_{path.stem}.png", dpi=160); plt.close(fig)


def plot_mel_filterbank(out: Path) -> None:
    filters = librosa.filters.mel(sr=FS, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=FS / 2)
    freqs = np.linspace(0, FS / 2, filters.shape[1])
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for filt in filters: ax.plot(freqs, filt, linewidth=.8)
    ax.set(title="Mel filterbank (24 bộ lọc)", xlabel="Tần số (Hz)", ylabel="Trọng số")
    ax.grid(alpha=.2); fig.savefig(out / "mel_filterbank.png", dpi=160); plt.close(fig)


def plot_mfcc(label: str, X: np.ndarray, out: Path, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.imshow(X.T, origin="lower", aspect="auto", interpolation="nearest",
              extent=[0, len(X) * HOP / FS, 0, X.shape[1]])
    ax.set(title=f"MFCC {VIETNAMESE.get(label, label)}{suffix}", xlabel="Thời gian (s)", ylabel="Hệ số")
    fig.savefig(out / f"mfcc_{label}{suffix.replace(' ', '_')}.png", dpi=160); plt.close(fig)


def plot_dtw(X: np.ndarray, Y: np.ndarray, title: str, filename: str, out: Path) -> float:
    score, path, local = dtw_distance(X, Y)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.imshow(local.T, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
    pi, pj = np.asarray(path).T
    ax.plot(pi, pj, color="cyan", linewidth=1.4)
    ax.set(title=f"{title} — DTW_norm={score:.3f}", xlabel="Frame X", ylabel="Frame Y")
    fig.savefig(out / filename, dpi=160); plt.close(fig)
    return score


def split_files(data: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    train, test = {}, {}
    for lab in LABELS:
        files = sorted((data / lab).glob("*.wav"))
        if len(files) < 5:
            raise ValueError(f"{data / lab}: cần tối thiểu 5 WAV, hiện có {len(files)}")
        train[lab], test[lab] = files[:3], files[3:]
    return train, test


def make_templates(files: dict[str, list[Path]], trim: bool, delta: bool = False):
    bank: dict[str, list[np.ndarray]] = {lab: [] for lab in LABELS}
    for lab, paths in files.items():
        for path in paths:
            y = load_audio(path)
            if trim: y, _ = trim_energy(y)
            bank[lab].append(mfcc_feature(y, use_delta=delta))
    return bank


def score_one(path: Path, templates, trim: bool, delta: bool = False):
    y = load_audio(path)
    if trim: y, _ = trim_energy(y)
    X = mfcc_feature(y, use_delta=delta)
    scores = {lab: min(dtw_distance(X, ref)[0] for ref in refs)
              for lab, refs in templates.items() if refs}
    ordered = sorted(scores.items(), key=lambda item: item[1])
    return ordered, X


def evaluate(test: dict[str, list[Path]], templates, trim: bool, delta: bool = False,
             print_top: bool = False):
    truth, pred, rows = [], [], []
    for lab in LABELS:
        for path in test[lab]:
            ordered, _ = score_one(path, templates, trim, delta)
            guessed = ordered[0][0]
            truth.append(lab); pred.append(guessed)
            rows.append((path.name, lab, guessed, ordered[0][1], ordered[1][1] if len(ordered) > 1 else ""))
            if print_top:
                print(f"{path.name:24s} thật={lab:5s} dự đoán={guessed:5s} | top-3: {ordered[:3]}")
    return accuracy_score(truth, pred), truth, pred, rows


def save_confusion(truth, pred, name: str, out: Path) -> None:
    cm = confusion_matrix(truth, pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(cm, display_labels=[VIETNAMESE[x] for x in LABELS])
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(name); plt.tight_layout(); plt.savefig(out / f"confusion_{name.replace(' ', '_')}.png", dpi=160); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--unknown", type=Path, help="Tùy chọn: WAV ngoài bộ dữ liệu để nhận dạng")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train, test = split_files(args.data)

    # Data quality checks and required waveform/time-feature figures.
    plot_waveform_examples(args.data, args.out)
    plot_mel_filterbank(args.out)
    for lab in LABELS[:3]:
        path = sorted((args.data / lab).glob("*.wav"))[0]
        y = load_audio(path)
        peak = float(np.max(np.abs(y)))
        print(f"Chất lượng {path}: {len(y)/FS:.2f}s, peak={peak:.3f}, clipping={np.mean(np.abs(y) >= .999)*100:.2f}%")
        plot_time_analysis(path, y, args.out)
        if lab == LABELS[0]: plot_autocorrelation(path, y, args.out)
        trimmed, bounds = trim_energy(y)
        sf.write(args.out / f"trim_{path.stem}.wav", trimmed, FS)
        print(f"  trim: {len(y)/FS:.2f}s -> {len(trimmed)/FS:.2f}s; samples={bounds}")

    # Baseline: endpoint detection + 13 MFCC.
    templates = make_templates(train, trim=True)
    for lab in LABELS[:2]:
        _, X = score_one(train[lab][0], templates, trim=True)
        plot_mfcc(lab, X, args.out)
    same_x = templates[LABELS[0]][0]
    same_y = templates[LABELS[0]][1]
    diff_y = templates[LABELS[1]][0]
    same_score = plot_dtw(same_x, same_y, "Cùng từ", "dtw_same.png", args.out)
    diff_score = plot_dtw(same_x, diff_y, "Khác từ", "dtw_different.png", args.out)
    print(f"DTW cùng từ={same_score:.3f}; khác từ={diff_score:.3f}")

    acc, truth, pred, rows = evaluate(test, templates, trim=True, print_top=True)
    print(f"Accuracy baseline (trim + 13 MFCC): {acc:.3f}")
    save_confusion(truth, pred, "trim_mfcc13", args.out)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["filename", "true_label", "predicted_label", "top1_dtw", "top2_dtw"]); writer.writerows(rows)

    # E1: with vs without endpoint detection; E2: 13 MFCC vs 13 MFCC + delta.
    no_trim_templates = make_templates(train, trim=False)
    acc_no, t_no, p_no, _ = evaluate(test, no_trim_templates, trim=False)
    delta_templates = make_templates(train, trim=True, delta=True)
    acc_delta, t_d, p_d, _ = evaluate(test, delta_templates, trim=True, delta=True)
    save_confusion(t_no, p_no, "no_trim_mfcc13", args.out)
    save_confusion(t_d, p_d, "trim_mfcc13_delta", args.out)
    with (args.out / "experiment_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["experiment", "accuracy"])
        writer.writerows([["E1 no endpoint + MFCC13", f"{acc_no:.4f}"],
                          ["E1 endpoint + MFCC13", f"{acc:.4f}"],
                          ["E2 endpoint + MFCC13", f"{acc:.4f}"],
                          ["E2 endpoint + MFCC13+delta", f"{acc_delta:.4f}"]])
    print(f"E1 accuracy: no-trim={acc_no:.3f}, trim={acc:.3f}")
    print(f"E2 accuracy: MFCC13={acc:.3f}, MFCC13+delta={acc_delta:.3f}")

    if args.unknown:
        ordered, _ = score_one(args.unknown, templates, trim=True)
        print(f"Unknown input {args.unknown}: dự đoán={ordered[0][0]} ({VIETNAMESE[ordered[0][0]]}), DTW_norm={ordered[0][1]:.3f}")
        print("Top-3:", ordered[:3])
    print(f"Đã lưu biểu đồ và CSV trong: {args.out.resolve()}")


if __name__ == "__main__":
    main()
