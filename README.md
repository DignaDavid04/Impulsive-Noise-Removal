# Impulsive Noise Removal

A web application for removing impulsive noise (clicks, pops, crackle) from music recordings using an adaptive Autoregressive (AR) model with Exponentially Weighted Least Squares (EW-LS) parameter estimation.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Features

| Feature | Description |
|:--------|:------------|
| **Drag & Drop Upload** | Simply drag your audio file into the browser |
| **Multi-Format Support** | WAV, MP3, OGG, FLAC, AAC, M4A, WMA, OPUS |
| **Real-Time Progress** | Live progress updates via WebSocket |
| **Audio Comparison** | Listen to original vs. cleaned audio side-by-side |
| **Analysis Plots** | Waveform, prediction error, AR coefficient evolution |
| **Adjustable Parameters** | Fine-tune AR order, threshold (η), and forgetting factor (λ) |
| **One-Click Download** | Download the cleaned WAV file instantly |

---

## How It Works

The algorithm implements a 4-stage pipeline:


1. **AR Modeling** — Models the audio signal as a 4th-order autoregressive process
2. **EW-LS Estimation** — Adaptively estimates AR coefficients using recursive least squares with exponential forgetting (λ = 0.99)
3. **Click Detection** — Identifies impulsive noise when prediction error exceeds an adaptive threshold (η · σₑ)
4. **Linear Interpolation** — Replaces corrupted samples by interpolating between surrounding clean samples

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **FFmpeg** (required for MP3/FLAC/AAC support via pydub)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### Installation

```bash

# Clone the repository
git clone https://github.com/DignaDavid04/Impulsive-Noise-Removal.git
cd Impulsive-Noise-Removal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```
Open your browser and navigate to http://localhost:5000

## Parameters

| Parameter | Symbol | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **AR Order** | r | 4 | 1-20 | Order of the autoregressive model |
| **Detection Threshold** | η | 3.5 | 0.5-10 | Sensitivity multiplier (lower = more sensitive) |
| **Forgetting Factor** | λ | 0.99 | 0.9-0.999 | Adaptation speed (closer to 1 = slower, more stable) |

---


### Project Structure

```text
Impulsive-Noise-Removal/
├── app.py                 # Flask web application & WebSocket server
├── noise_remover.py       # Core noise removal algorithm
├── requirements.txt       # Python dependencies
├── License
├── README.md
├── static/
│   ├── css/
│   │   └── style.css      # Application styles
│   └── js/
│       └── app.js         # Frontend JavaScript
├── templates/
│   └── index.html         # Main HTML template
├── uploads/               # Temporary upload storage
└── outputs/               # Cleaned audio output storage
```
---

## Algorithm Details

Based on the methodology from **System Identification** (Maciej Niedźwiecki, Gdańsk University of Technology):

* **Signal Model: AR(r):** y(t) = Σᵢ aᵢ·y(t-i) + n(t)
* **Prediction Error:** ε(t) = y(t) - ŷ(t|t-1)
* **Variance Estimate:** σ̂²ₑ(t) = λ·σ̂²ₑ(t-1) + (1-λ)·ε²(t)
* **Detection Rule:** d̂(t) = 1 if |ε(t)| > η·σ̂ₑ(t)
* **Correction:** Linear interpolation between boundary clean samples

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Flask, Flask-SocketIO |
| **Algorithm** | NumPy, SciPy, Matplotlib |
| **Audio I/O** | scipy.io.wavfile, pydub |
| **Frontend** | Vanilla JS, CSS3, Canvas API |
| **Real-time** | WebSocket (Socket.IO) |

---

## License

**MIT License** — feel free to use, modify, and distribute.

---

## Author

**Digna David** (206959)  
**Project:** SI Project - Removal of Impulsive Noise from Music Recordings  
**University:** Gdańsk University of Technology  
**Supervisor:** Piotr Kaczmarek

---

## References

[1] Maciej Niedźwiecki: "System Identification" — Lecture Notes, Gdańsk University of Technology, Department of Automatic Control, Gdańsk, Poland.

