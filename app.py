"""
Q3B — Audio Fingerprint Identifier
Streamlit app that identifies a song from an uploaded audio clip
using the fingerprint database built in Q3A.
"""

import os
import gc
import wave
import pickle
import tempfile
import collections

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
from pydub import AudioSegment

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
    seg     = AudioSegment.from_file(path)
    seg     = seg.set_channels(1).set_frame_rate(target_fs)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * seg.sample_width - 1))
    del seg
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
    # Overlay peaks
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
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#2ecc71", label="Winner"),
                 Patch(color="#3498db", label="Other")],
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
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎵 Audio Fingerprint Identifier")
st.caption("Q3B — EE200 | Shazam-style song recognition via constellation-map hashing")

database, song_index, db_params = load_database(DB_PATH)

# ── Sidebar: database info ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Database")
    if database is None:
        st.error(f"Database not found at `{DB_PATH}`.\nRun Q3A Phases 1–5 first.")
    else:
        st.success(f"{len(song_index)} songs indexed")
        st.json({
            "window":      db_params["window"],
            "overlap":     db_params["overlap"],
            "n_peaks":     db_params["n_peaks"],
            "fan_out":     db_params["fan_out"],
            "target_fs":   db_params["target_fs"],
        })
        st.subheader("Indexed songs")
        for sid in sorted(song_index):
            st.write(f"• {song_index[sid]}")

# ── Main: file upload ─────────────────────────────────────────────────────────
if database is None:
    st.stop()

uploaded = st.file_uploader(
    "Upload an audio clip to identify",
    type=[ext.lstrip(".") for ext in AUDIO_EXTENSIONS],
    help="Short clips (5–30 s) work best. The clip must come from one of the indexed songs.",
)

if uploaded is None:
    st.info("⬆️ Upload an audio file to get started.")
    st.stop()

# ── Process uploaded file ─────────────────────────────────────────────────────
with st.spinner("Decoding audio…"):
    suffix = os.path.splitext(uploaded.name)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        fs, audio = load_audio(tmp_path)
    except Exception as e:
        st.error(f"Could not decode audio: {e}")
        os.unlink(tmp_path)
        st.stop()
    finally:
        os.unlink(tmp_path)

duration = len(audio) / fs

st.subheader("Uploaded clip")
col1, col2, col3 = st.columns(3)
col1.metric("Duration",     f"{duration:.2f} s")
col2.metric("Sample rate",  f"{fs} Hz")
col3.metric("Samples",      f"{len(audio):,}")

# Waveform
st.pyplot(plot_waveform(audio, fs, title=f"Waveform — {uploaded.name}"))

# ── Fingerprint + match ───────────────────────────────────────────────────────
with st.spinner("Fingerprinting and matching…"):
    freqs, times, Sxx_db, peaks, hashes, ranked, total_hits = identify(
        audio, fs, database, song_index
    )

# Spectrogram with peaks
st.pyplot(plot_spectrogram(
    times, freqs, Sxx_db, peaks,
    title=f"Spectrogram + Constellation Map — {uploaded.name}",
))

st.subheader("Fingerprint stats")
c1, c2, c3 = st.columns(3)
c1.metric("Peaks extracted", len(peaks))
c2.metric("Hashes generated", len(hashes))
c3.metric("Total DB hits", total_hits)

# ── Results ───────────────────────────────────────────────────────────────────
st.subheader("Identification result")

if not ranked:
    st.error("No matching hashes found. The clip may not be in the database.")
    st.stop()

winner_id,   winner_votes   = ranked[0]
winner_name                 = song_index[winner_id]
confidence                  = 100.0 * winner_votes / max(total_hits, 1)

if len(ranked) >= 2:
    runner_id,   runner_votes = ranked[1]
    runner_name               = song_index[runner_id]
    margin                    = winner_votes - runner_votes
else:
    runner_name  = "—"
    runner_votes = 0
    margin       = winner_votes

# Result banner
st.success(f"### 🎶 {winner_name.replace('_', ' ')}")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Winner votes",   winner_votes)
r2.metric("Runner-up",      runner_name.replace("_", " "), delta=f"−{margin} votes")
r3.metric("Runner-up votes", runner_votes)
r4.metric("Confidence",     f"{confidence:.1f}%")

# Vote chart
st.pyplot(plot_votes(ranked, song_index))

# Full ranked table
with st.expander("Full vote table"):
    rows = []
    for rank, (sid, v) in enumerate(ranked, 1):
        rows.append({
            "Rank":  rank,
            "Song":  song_index[sid],
            "Votes": v,
            "% of hits": f"{100*v/max(total_hits,1):.1f}%",
        })
    st.table(rows)
