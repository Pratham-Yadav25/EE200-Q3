# EE200-Q3: Audio Fingerprint Identifier

## Project Overview

This project implements a **Shazam-style Audio Fingerprinting System** for automatic song identification. The system extracts unique fingerprints from audio signals using spectrogram analysis and landmark hashing, then matches them against a pre-built fingerprint database.

The implementation demonstrates concepts from **Signals and Systems**, including time-frequency analysis, spectral peak detection, and pattern matching.

---

## Features

### Single Clip Identification

* Upload an audio file (MP3, WAV, FLAC, M4A, OGG)
* Generate waveform visualization
* Compute spectrogram using STFT
* Extract constellation-map peaks
* Generate audio fingerprints (landmark hashes)
* Match against fingerprint database
* Display confidence score and ranked matches

### Visualizations

* Waveform
* Spectrogram
* Constellation Map
* Offset Histogram
* Match Score Chart

### Batch Processing

* Upload multiple audio files simultaneously
* Automatic identification of all clips
* Batch results table
* Download results as CSV

---

## Audio Fingerprinting Pipeline

1. Load audio clip
2. Compute STFT spectrogram
3. Detect spectral peaks
4. Generate constellation map
5. Create landmark hashes
6. Search fingerprint database
7. Build offset histogram
8. Select highest-vote song match

---

## Fingerprint Database

Database statistics:

* Songs indexed: **50**
* Unique hash keys: **29,195**
* STFT window size: **4096**
* Overlap: **2048**
* Fan-out: **15**
* Sample rate: **22050 Hz**

---

## Technology Stack

* Python
* NumPy
* SciPy
* Librosa
* Pandas
* Matplotlib
* Streamlit

---

## Project Structure

```text
EE200-Q3/
│
├── app.py
├── requirements.txt
├── README.md
│
└── outputs/
    └── fingerprint_database.pkl
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd EE200-Q3
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Results

The system successfully identifies songs from a database of 50 tracks and demonstrates robustness against:

* Additive noise
* Time-shifted queries
* Partial song clips

The implementation follows the landmark-hash fingerprinting approach used in practical music recognition systems such as Shazam.

---

## Deployment

### Streamlit Application

Add your deployed Streamlit URL here:

```text
https://ee200-q3.streamlit.app
```

### GitHub Repository

Add your GitHub repository URL here:

```text
https://github.com/Pratham-Yadav25/EE200-Q3
```

---

## Course Information

**Course:** EE200 – Signals and Systems

**Project:** Audio Fingerprinting and Song Identification

**Institute:** Indian Institute of Technology Kanpur (IIT Kanpur)
