# EE200-Q3: Audio Fingerprint Identifier

## Project Overview

This project implements a Shazam-style audio fingerprinting system for song identification.

The system uses:

* Short-Time Fourier Transform (STFT)
* Spectrogram analysis
* Constellation-map peak extraction
* Hash-based audio fingerprinting
* Database matching using landmark hashes

A Streamlit web application is provided for interactive song recognition.

---

## Features

### Single Clip Identification

* Upload an audio file
* Generate spectrogram
* Extract constellation-map peaks
* Create audio fingerprints
* Match against fingerprint database
* Display confidence score and ranked matches

### Visualizations

* Waveform
* Spectrogram
* Constellation map
* Offset histogram
* Match score chart

### Batch Processing

* Upload multiple audio files
* Identify all clips automatically
* Generate results table
* Download results as CSV

---

## Database

Fingerprint database contains:

* 50 indexed songs
* 29,195 unique hash keys
* STFT window size: 4096
* Overlap: 2048
* Fan-out: 15

---

## Technology Stack

* Python
* NumPy
* SciPy
* Librosa
* Matplotlib
* Pandas
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

Clone the repository:

```bash
git clone <repository-url>
cd EE200-Q3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

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

## Results

The system successfully identifies songs from a database of 50 tracks and demonstrates robustness against:

* Additive noise
* Time-shifted queries
* Partial song clips

The implementation follows the landmark-hash fingerprinting approach used in practical music recognition systems.

---

## Course

EE200 – Signals and Systems

Course Project: Audio Fingerprinting and Song Identification
