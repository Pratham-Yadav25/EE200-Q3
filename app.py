"""
Q3B — Audio Fingerprint Identifier
Streamlit app that identifies a song from an uploaded audio clip
using the fingerprint database built in Q3A.
"""

import os
import gc
import io
import wave
import pickle
import tempfile
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
import soundfile as sf
import resampy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Fingerprint Identifier",
    page_icon="🎵",
    layout="wide",
)

# ── Constants (must match Q3A database construction exactly) ──────────────────
TARGET_FS   = 22050
WIN         = 4096
OVERLAP     = 2048
FILTER_SIZE = 20
N_PEAKS     = 500
FAN_OUT     = 15
DB_PATH     = "outputs/fingerprint_database.pkl"

AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]


# ══════════════════════════════════════════════════════════════════════════════
# Core fingerprinting functions  (identical to Q3A notebook helpers)
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path: str, target_fs: int = TARGET_FS):
    """Decode any supported audio format → mono float32 resampled to target_fs."""
    samples, fs = sf.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if fs != target_fs:
        samples = resampy.resample(samples, fs, target_fs)
    gc.collect()
    return target_fs, np.clip(samples, -1.0, 1.0)


def get_spectrogram(audio: np.ndarray, fs: int,
                    win: int = WIN, overlap: int = OVERLAP):
    """Return (freqs, times, Sxx_db) all as float32."""
    f, t, S = signal.spectrogram(
        audio.astype(np.float32), fs=fs,
        window="hann", nperseg=win, noverlap=overlap,
    )
    Sxx_db = (10 * np.log10(S.astype(np.float32) + 1e-10)).astype(np.float32)
    del S
    gc.collect()
    return f, t, Sxx_db


def get_peaks(Sxx_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
              filter_size: int = FILTER_SIZE, n_keep: int = N_PEAKS):
    """Return peak array columns: [freq_bin, time_bin, freq_hz, time_s, amp_db]."""
    lm      = Sxx_db == maximum_filter(Sxx_db, size=filter_size)
    fi, ti  = np.where(lm)
    amps    = Sxx_db[fi, ti]
    top     = np.argsort(amps)[::-1][:n_keep]
    fi, ti, amps = fi[top], ti[top], amps[top]
    s = np.argsort(ti)
    return np.column_stack([fi[s], ti[s], freqs[fi[s]], times[ti[s]], amps[s]])


def get_hashes(peaks: np.ndarray, fan_out: int = FAN_OUT):
    """Return list of (hash_value, anchor_time_bin)."""
    out, n = [], len(peaks)
    for i in range(n):
        f1, t1 = int(peaks[i, 0]), int(peaks[i, 1])
        for j in range(i + 1, min(i + 1 + fan_out, n)):
            f2 = int(peaks[j, 0])
            dt = int(peaks[j, 1]) - t1
            h  = ((f1 & 0x7FF) << 42) | ((f2 & 0x7FF) << 31) | (dt & 0x7FFFFFFF)
            out.append((h, t1))
    return out


def identify(audio: np.ndarray, fs: int, database: dict, song_index: dict):
    """Full pipeline: audio → spectrogram → peaks → hashes → votes → result."""
    freqs, times, Sxx_db = get_spectrogram(audio, fs)
    peaks                = get_peaks(Sxx_db, freqs, times)
    hashes               = get_hashes(peaks)

    votes      = collections.Counter()
    total_hits = 0
    for h_val, _ in hashes:
        if h_val in database:
            for sid, _ in database[h_val]:
                votes[sid] += 1
                total_hits += 1

    ranked = votes.most_common()
    return freqs, times, Sxx_db, peaks, hashes, ranked, total_hits


def compute_offsets(hashes: list, winner_id: int, database: dict):
    """
    For the winning song, compute time offsets between each matching
    database anchor time and the corresponding query anchor time.
    offset = db_anchor_time_bin - query_anchor_time_bin
    Returns a list of integer offsets.
    """
    offsets = []
    for h_val, q_t in hashes:
        if h_val in database:
            for sid, db_t in database[h_val]:
                if sid == winner_id:
                    offsets.append(int(db_t) - int(q_t))
    return offsets


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_waveform(audio: np.ndarray, fs: int, title: str = "Waveform"):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    t = np.arange(len(audio)) / fs
    ax.plot(t, audio, linewidth=0.5, color="#3498db")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig


def plot_spectrogram(times, freqs, Sxx_db, peaks, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        Sxx_db, aspect="auto", origin="lower", cmap="viridis",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    ax.scatter(
        times[peaks[:, 1].astype(int)],
        freqs[peaks[:, 0].astype(int)],
        color="red", s=25, zorder=5, linewidths=0.5,
        edgecolors="white", label=f"Top {len(peaks)} peaks",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_offset_histogram(offsets: list, winner_name: str):
    """
    Plot histogram of time offsets for the winning song.
    Highlights the peak bin in green.
    """
    if not offsets:
        return None, None, None

    offsets_arr = np.array(offsets)
    # Bin width = 1 (one time-frame unit)
    bin_min = int(offsets_arr.min())
    bin_max = int(offsets_arr.max())
    bins = np.arange(bin_min, bin_max + 2) - 0.5  # edges centred on integers

    fig, ax = plt.subplots(figsize=(10, 3.5))
    counts, edges, patches = ax.hist(
        offsets_arr, bins=bins, color="#3498db", edgecolor="white", linewidth=0.4
    )

    # Find and highlight peak bin
    peak_idx   = int(np.argmax(counts))
    peak_count = int(counts[peak_idx])
    peak_offset = int(round((edges[peak_idx] + edges[peak_idx + 1]) / 2))

    patches[peak_idx].set_facecolor("#2ecc71")
    patches[peak_idx].set_edgecolor("white")

    ax.axvline(peak_offset, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Peak offset = {peak_offset}  (count = {peak_count})")

    ax.set_xlabel("Offset  (DB anchor time − query anchor time)  [frames]")
    ax.set_ylabel("Matching hashes (votes)")
    ax.set_title(f"Offset Histogram — {winner_name.replace('_', ' ')}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, peak_offset, peak_count


def plot_votes(ranked, song_index, top_n=10):
    top    = ranked[:top_n]
    labels = [song_index[sid].replace(".mp3","").replace(".wav","")
                             .replace("_"," ")
              for sid, _ in top]
    scores = [v for _, v in top]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(top)), scores, color=colors,
                  edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, str(score),
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Votes (matching hashes)")
    ax.set_title("Top Match Scores")
    ax.legend(
        handles=[mpatches.Patch(color="#2ecc71", label="Winner"),
                 mpatches.Patch(color="#3498db", label="Other")],
        loc="upper right", fontsize=8,
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Load database (cached so it is read only once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading fingerprint database…")
def load_database(path: str):
    if not os.path.exists(path):
        return None, None, None
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["database"], payload["song_index"], payload["params"]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helper: process one uploaded file object → prediction string
# ══════════════════════════════════════════════════════════════════════════════

def process_file_object(uploaded_file, database, song_index):
    """
    Load, fingerprint, and identify a single Streamlit UploadedFile.
    Returns (fs, audio, freqs, times, Sxx_db, peaks, hashes, ranked, total_hits).
    Raises on decode failure.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        fs, audio = load_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    freqs, times, Sxx_db, peaks, hashes, ranked, total_hits = identify(
        audio, fs, database, song_index
    )
    return fs, audio, freqs, times, Sxx_db, peaks, hashes, ranked, total_hits


# ══════════════════════════════════════════════════════════════════════════════
# UI — title & database load
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎵 Audio Fingerprint Identifier")
st.caption("Q3B — EE200 | Shazam-style song recognition via constellation-map hashing")

database, song_index, db_params = load_database(DB_PATH)

# ── Sidebar: enhanced database info panel ─────────────────────────────────────
with st.sidebar:
    st.header("📂 Database Info")
    if database is None:
        st.error(f"Database not found at `{DB_PATH}`.\nRun Q3A Phases 1–5 first.")
    else:
        n_songs  = len(song_index)
        n_hashes = len(database)

        st.success(f"✅ Database loaded")

        st.markdown("#### Summary")
        m1, m2 = st.columns(2)
        m1.metric("Songs indexed", n_songs)
        m2.metric("Unique hashes", f"{n_hashes:,}")

        st.markdown("#### Parameters")
        params_display = {
            "Window size":  db_params.get("window",    WIN),
            "Overlap":      db_params.get("overlap",   OVERLAP),
            "Fan-out":      db_params.get("fan_out",   FAN_OUT),
            "# Peaks":      db_params.get("n_peaks",   N_PEAKS),
            "Sample rate":  db_params.get("target_fs", TARGET_FS),
        }
        for k, v in params_display.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("#### Indexed Songs")
        for sid in sorted(song_index):
            label = song_index[sid].replace(".mp3","").replace(".wav","").replace("_"," ")
            st.markdown(f"🎵 {label}")

# ── Guard ─────────────────────────────────────────────────────────────────────
if database is None:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🎧 Single Clip", "📂 Batch Mode"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Clip  (existing functionality, unchanged logic)
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    uploaded = st.file_uploader(
        "Upload an audio clip to identify",
        type=[ext.lstrip(".") for ext in AUDIO_EXTENSIONS],
        help="Short clips (5–30 s) work best. The clip must come from one of the indexed songs.",
        key="single_uploader",
    )

    if uploaded is None:
        st.info("⬆️ Upload an audio file to get started.")
        st.stop()

    # ── Decode ────────────────────────────────────────────────────────────────
    with st.spinner("Decoding audio…"):
        try:
            fs, audio, freqs, times, Sxx_db, peaks, hashes, ranked, total_hits = (
                process_file_object(uploaded, database, song_index)
            )
        except Exception as e:
            st.error(f"Could not decode audio: {e}")
            st.stop()

    duration = len(audio) / fs

    st.subheader("Uploaded clip")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration",    f"{duration:.2f} s")
    col2.metric("Sample rate", f"{fs} Hz")
    col3.metric("Samples",     f"{len(audio):,}")

    # Waveform
    st.pyplot(plot_waveform(audio, fs, title=f"Waveform — {uploaded.name}"))

    # Spectrogram + constellation map
    st.pyplot(plot_spectrogram(
        times, freqs, Sxx_db, peaks,
        title=f"Spectrogram + Constellation Map — {uploaded.name}",
    ))

    st.subheader("Fingerprint stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Peaks extracted",  len(peaks))
    c2.metric("Hashes generated", len(hashes))
    c3.metric("Total DB hits",    total_hits)

    # ── Offset histogram ──────────────────────────────────────────────────────
    if not ranked:
        st.error("No matching hashes found. The clip may not be in the database.")
        st.stop()

    winner_id, winner_votes = ranked[0]
    winner_name             = song_index[winner_id]

    st.subheader("⏱️ Offset Histogram")
    offsets = compute_offsets(hashes, winner_id, database)

    if offsets:
        fig_hist, peak_offset, peak_count = plot_offset_histogram(offsets, winner_name)
        st.pyplot(fig_hist)

        oh1, oh2, oh3 = st.columns(3)
        oh1.metric("Total offset votes", len(offsets))
        oh2.metric("Peak offset (frames)", peak_offset)
        oh3.metric("Peak bin count", peak_count)
    else:
        st.warning("No offset data available for the winning song.")

    # ── Final identification result ───────────────────────────────────────────
    st.subheader("Identification result")

    confidence = 100.0 * winner_votes / max(total_hits, 1)

    if len(ranked) >= 2:
        runner_id,   runner_votes = ranked[1]
        runner_name               = song_index[runner_id]
        margin                    = winner_votes - runner_votes
    else:
        runner_name  = "—"
        runner_votes = 0
        margin       = winner_votes

    st.success(f"### 🎶 {winner_name.replace('_', ' ')}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Winner votes",    winner_votes)
    r2.metric("Runner-up",       runner_name.replace("_", " "), delta=f"−{margin} votes")
    r3.metric("Runner-up votes", runner_votes)
    r4.metric("Confidence",      f"{confidence:.1f}%")

    # Vote chart
    st.pyplot(plot_votes(ranked, song_index))

    # Full ranked table
    with st.expander("Full vote table"):
        rows = []
        for rank, (sid, v) in enumerate(ranked, 1):
            rows.append({
                "Rank":       rank,
                "Song":       song_index[sid],
                "Votes":      v,
                "% of hits":  f"{100*v/max(total_hits,1):.1f}%",
            })
        st.table(rows)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Mode
# ════════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("📂 Batch Identification")
    st.markdown(
        "Upload **multiple audio clips** at once. Each file will be fingerprinted "
        "and matched against the database. Results can be downloaded as `results.csv`."
    )

    batch_files = st.file_uploader(
        "Upload audio clips",
        type=[ext.lstrip(".") for ext in AUDIO_EXTENSIONS],
        accept_multiple_files=True,
        help="Select or drag in multiple files.",
        key="batch_uploader",
    )

    if not batch_files:
        st.info("⬆️ Upload one or more audio files to run batch identification.")
    else:
        batch_results = []   # list of dicts: filename, prediction, votes, confidence

        progress_bar = st.progress(0, text="Starting…")
        status_box   = st.empty()

        for i, f in enumerate(batch_files):
            status_box.markdown(f"🔍 Processing **{f.name}** ({i+1}/{len(batch_files)})…")
            try:
                _, _, _, _, _, _, batch_hashes, batch_ranked, batch_hits = (
                    process_file_object(f, database, song_index)
                )
                if batch_ranked:
                    pred_id, pred_votes = batch_ranked[0]
                    pred_name  = song_index[pred_id]
                    pred_conf  = 100.0 * pred_votes / max(batch_hits, 1)
                else:
                    pred_name  = "No match"
                    pred_votes = 0
                    pred_conf  = 0.0
            except Exception as exc:
                pred_name  = f"ERROR: {exc}"
                pred_votes = 0
                pred_conf  = 0.0

            batch_results.append({
                "filename":   f.name,
                "prediction": pred_name.rsplit(".", 1)[0],,
                "votes":      pred_votes,
                "confidence": f"{pred_conf:.1f}%",
            })
            progress_bar.progress((i + 1) / len(batch_files),
                                  text=f"Done {i+1}/{len(batch_files)}")

        status_box.success(f"✅ Batch complete — {len(batch_results)} files processed.")

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader("Batch Results")
        df_display = pd.DataFrame(batch_results)
        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "filename":   st.column_config.TextColumn("File"),
                "prediction": st.column_config.TextColumn("Prediction"),
                "votes":      st.column_config.NumberColumn("Votes", format="%d"),
                "confidence": st.column_config.TextColumn("Confidence"),
            },
            hide_index=True,
        )

        # ── CSV export ────────────────────────────────────────────────────────
        # Required format: filename,prediction
        df_export = df_display[["filename", "prediction"]]
        csv_bytes  = df_export.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            help="Downloads a CSV with columns: filename, prediction",
        )

        # Per-file confidence summary chart
        if len(batch_results) > 1:
            st.subheader("Batch confidence overview")
            conf_vals  = [float(r["confidence"].rstrip("%")) for r in batch_results]
            file_labels = [r["filename"] for r in batch_results]

            fig_b, ax_b = plt.subplots(figsize=(max(6, len(batch_results)*1.2), 3.5))
            bar_colors = ["#2ecc71" if c >= 50 else "#e67e22" if c >= 20 else "#e74c3c"
                          for c in conf_vals]
            ax_b.bar(range(len(conf_vals)), conf_vals, color=bar_colors,
                     edgecolor="white", linewidth=0.8)
            ax_b.set_xticks(range(len(file_labels)))
            ax_b.set_xticklabels(file_labels, rotation=30, ha="right", fontsize=8)
            ax_b.set_ylabel("Confidence (%)")
            ax_b.set_ylim(0, max(conf_vals) * 1.15 + 1)
            ax_b.set_title("Per-file Identification Confidence")
            ax_b.axhline(50, color="#bdc3c7", linestyle="--", linewidth=0.8, label="50% threshold")
            for idx, (val, bar) in enumerate(zip(conf_vals,
                                                  ax_b.patches)):
                ax_b.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.3,
                          f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
            ax_b.legend(
                handles=[
                    mpatches.Patch(color="#2ecc71", label="High (≥50%)"),
                    mpatches.Patch(color="#e67e22", label="Medium (20–49%)"),
                    mpatches.Patch(color="#e74c3c", label="Low (<20%)"),
                ],
                fontsize=7, loc="upper right",
            )
            fig_b.tight_layout()
            st.pyplot(fig_b)
