"""
Impulsive Noise Removal Engine
Uses an adaptive AR model with EW-LS for click/pop detection and linear interpolation for removal.
"""

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import warnings
import os
import io
import base64
import tempfile

warnings.filterwarnings("ignore", category=DeprecationWarning)


def convert_to_wav(input_filepath):
    """
    Convert any supported audio format to WAV using pydub.
    Returns the path to the WAV file (original path if already WAV).
    """
    ext = os.path.splitext(input_filepath)[1].lower()

    if ext == '.wav':
        return input_filepath, False  # Already WAV, no temp file created

    supported_formats = {
        '.mp3': 'mp3',
        '.ogg': 'ogg',
        '.flac': 'flac',
        '.aac': 'aac',
        '.m4a': 'mp4',
        '.wma': 'wma',
        '.opus': 'opus',
    }

    if ext not in supported_formats:
        raise ValueError(f"Unsupported audio format: {ext}")

    audio = AudioSegment.from_file(input_filepath, format=supported_formats[ext])
    wav_path = tempfile.mktemp(suffix='.wav')
    audio.export(wav_path, format='wav')
    return wav_path, True  # Return path and flag that it's a temp file


def remove_impulsive_noise(input_filepath, output_filepath, ar_rank=4, eta=3.0,
                           forgetting_factor=0.99, progress_callback=None):
    """
    Removes impulsive noise from an audio file using an adaptive AR model
    with EW-LS parameter estimation and linear interpolation correction.

    Args:
        input_filepath (str): Path to the input audio file.
        output_filepath (str): Path to save the cleaned WAV file.
        ar_rank (int): The rank (order) of the AR model.
        eta (float): The threshold multiplier for noise detection (typically 1.5 to 4).
        forgetting_factor (float): The forgetting factor (lambda) for the EW-LS algorithm.
        progress_callback (callable): Optional callback function(progress_percent, message).

    Returns:
        dict: Results containing statistics and plot data.
    """

    def update_progress(percent, message="Processing..."):
        if progress_callback:
            progress_callback(percent, message)

    update_progress(5, "Reading audio file...")

    # Convert to WAV if necessary
    temp_wav_created = False
    try:
        wav_filepath, temp_wav_created = convert_to_wav(input_filepath)
        samplerate, data = wavfile.read(wav_filepath)
        original_dtype = data.dtype
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_filepath}")
    except Exception as e:
        raise RuntimeError(f"Error reading audio file: {str(e)}")
    finally:
        if temp_wav_created and os.path.exists(wav_filepath):
            os.remove(wav_filepath)

    update_progress(10, "Pre-processing audio...")

    # --- 1. Pre-processing ---
    if data.ndim > 1:
        audio = data.mean(axis=1)
    else:
        audio = data.astype(float)

    max_abs = np.max(np.abs(audio))
    if max_abs == 0:
        max_abs = 1.0
    audio_norm = audio / max_abs
    n_samples = len(audio_norm)
    cleaned_audio = np.copy(audio_norm)

    duration_seconds = n_samples / samplerate

    update_progress(15, "Initializing AR model parameters...")

    # --- 2. Initialization for EW-LS ---
    r = ar_rank
    lam = forgetting_factor
    a = np.zeros(r)
    P = np.eye(r) * 1000
    error_var_estimate = 0.01

    # Data collection for plots
    prediction_errors = []
    detection_thresholds = []
    ar_coeffs_history = [[] for _ in range(r)]
    detected_clicks_indices = []

    update_progress(20, "Processing audio samples...")

    # --- 3. Main Processing Loop ---
    k = r
    total_steps = n_samples - (r + 1) - r
    last_progress = 20

    while k < n_samples - (r + 1):
        # Progress update (map k to 20-80% range)
        current_progress = 20 + int(60 * (k - r) / max(total_steps, 1))
        if current_progress > last_progress + 2:
            update_progress(current_progress, f"Processing sample {k:,} / {n_samples:,}...")
            last_progress = current_progress

        y_past = np.flip(cleaned_audio[k - r:k])
        y_pred = np.dot(a, y_past)
        error = audio_norm[k] - y_pred

        error_var_estimate = lam * error_var_estimate + (1 - lam) * error ** 2
        delta_e = np.sqrt(error_var_estimate)
        threshold = eta * delta_e

        prediction_errors.append(error)
        detection_thresholds.append(threshold)

        # --- 4. Noise Detection and Correction ---
        if abs(error) > threshold:
            detected_clicks_indices.append(k)
            click_start = k
            click_end = k

            for j in range(1, r):
                if k + j >= n_samples - 1:
                    break
                next_y_past = np.flip(cleaned_audio[k + j - r:k + j])
                next_y_pred = np.dot(a, next_y_past)
                next_error = audio_norm[k + j] - next_y_pred
                if abs(next_error) > threshold:
                    click_end = k + j
                else:
                    break

            # --- 5. Linear Interpolation ---
            num_bad_samples = click_end - click_start + 1
            if click_start > 0 and click_end < n_samples - 1:
                y_start_good = cleaned_audio[click_start - 1]
                y_end_good = audio_norm[click_end + 1]
                for i in range(num_bad_samples):
                    idx_to_replace = click_start + i
                    alpha = (i + 1) / (num_bad_samples + 1)
                    cleaned_audio[idx_to_replace] = (1 - alpha) * y_start_good + alpha * y_end_good

            k = click_end + 1
            continue

        # --- 6. EW-LS Parameter Update ---
        g_num = P @ y_past
        g_den = lam + y_past.T @ g_num
        g = g_num / g_den
        P = (1 / lam) * (P - np.outer(g, y_past) @ P)
        a = a + g * error

        for i in range(r):
            ar_coeffs_history[i].append(a[i])

        k += 1

    update_progress(82, "Saving cleaned audio...")

    # --- 7. Post-processing ---
    cleaned_audio_denorm = cleaned_audio * max_abs
    cleaned_audio_final = cleaned_audio_denorm.astype(original_dtype)
    wavfile.write(output_filepath, samplerate, cleaned_audio_final)

    update_progress(88, "Generating analysis plots...")

    # --- 8. Generate Plots ---
    plots = generate_plots(
        audio_norm, cleaned_audio, prediction_errors, detection_thresholds,
        ar_coeffs_history, detected_clicks_indices, n_samples, samplerate, r
    )

    update_progress(95, "Compiling results...")

    # --- 9. Compile Results ---
    results = {
        'sample_rate': samplerate,
        'duration': round(duration_seconds, 2),
        'total_samples': n_samples,
        'clicks_detected': len(detected_clicks_indices),
        'clicks_per_second': round(len(detected_clicks_indices) / duration_seconds, 2) if duration_seconds > 0 else 0,
        'ar_order': r,
        'eta': eta,
        'lambda': forgetting_factor,
        'plots': plots,
    }

    update_progress(100, "Done!")
    return results


def generate_plots(audio_norm, cleaned_audio, prediction_errors, detection_thresholds,
                   ar_coeffs_history, detected_clicks_indices, n_samples, samplerate, r):
    """Generate base64-encoded plot images for the web UI."""

    plots = {}
    plt.style.use('seaborn-v0_8-darkgrid')

    # Determine a good zoom region centered on a detected click
    if detected_clicks_indices:
        center = detected_clicks_indices[len(detected_clicks_indices) // 2]
        plot_start = max(0, center - 1000)
        plot_end = min(n_samples, center + 1000)
    else:
        plot_start = max(0, n_samples // 2 - 1000)
        plot_end = min(n_samples, n_samples // 2 + 1000)

    time_axis = np.arange(n_samples) / samplerate

    # --- Plot 1: Waveform Comparison ---
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(time_axis[plot_start:plot_end], audio_norm[plot_start:plot_end],
             label='Original (Noisy)', color='#e74c3c', alpha=0.75, linewidth=0.8)
    ax1.plot(time_axis[plot_start:plot_end], cleaned_audio[plot_start:plot_end],
             label='Cleaned', color='#2980b9', alpha=0.75, linewidth=0.8)
    ax1.set_title('Waveform Comparison (Zoomed In)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    fig1.tight_layout()
    plots['waveform'] = fig_to_base64(fig1)
    plt.close(fig1)

    # --- Plot 2: Prediction Error & Threshold ---
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    time_axis_err = time_axis[r:r + len(prediction_errors)]
    if len(time_axis_err) == len(prediction_errors):
        ax2.plot(time_axis_err, prediction_errors, label='Prediction Error',
                 color='gray', alpha=0.5, linewidth=0.5)
        ax2.plot(time_axis_err, detection_thresholds, label='Threshold (+η·δₑ)',
                 color='#e67e22', linestyle='--', linewidth=1.2)
        ax2.plot(time_axis_err, -np.array(detection_thresholds), label='Threshold (−η·δₑ)',
                 color='#e67e22', linestyle='--', linewidth=1.2)

    if detected_clicks_indices:
        valid_times = []
        valid_values = []
        for idx in detected_clicks_indices:
            if idx - r >= 0 and idx - r < len(prediction_errors):
                valid_times.append(idx / samplerate)
                valid_values.append(prediction_errors[idx - r])
        ax2.scatter(valid_times, valid_values, color='#e74c3c', s=30,
                    zorder=5, label='Detected Clicks', alpha=0.7)

    if plot_start < plot_end:
        ax2.set_xlim(time_axis[plot_start], time_axis[plot_end])

    ax2.set_title('Prediction Error & Adaptive Detection Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error Amplitude')
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    plots['error'] = fig_to_base64(fig2)
    plt.close(fig2)

    # --- Plot 3: AR Coefficients ---
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']
    for i in range(len(ar_coeffs_history)):
        if ar_coeffs_history[i]:
            color = colors[i % len(colors)]
            ax3.plot(ar_coeffs_history[i], label=f'a₍{i + 1}₎', color=color,
                     linewidth=0.7, alpha=0.85)
    ax3.set_title('Evolution of AR Model Parameters', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Processing Step (Clean Samples)')
    ax3.set_ylabel('Coefficient Value')
    ax3.legend(loc='upper right')
    fig3.tight_layout()
    plots['ar_coeffs'] = fig_to_base64(fig3)
    plt.close(fig3)

    # --- Plot 4: Full Waveform Overview ---
    fig4, ax4 = plt.subplots(figsize=(14, 4))
    # Downsample for display performance
    downsample_factor = max(1, n_samples // 10000)
    display_orig = audio_norm[::downsample_factor]
    display_clean = cleaned_audio[::downsample_factor]
    display_time = time_axis[::downsample_factor]
    ax4.plot(display_time, display_orig, label='Original', color='#e74c3c', alpha=0.6, linewidth=0.5)
    ax4.plot(display_time, display_clean, label='Cleaned', color='#2980b9', alpha=0.6, linewidth=0.5)

    # Mark click regions
    if detected_clicks_indices:
        for idx in detected_clicks_indices[::max(1, len(detected_clicks_indices) // 50)]:
            ax4.axvline(x=idx / samplerate, color='#e74c3c', alpha=0.15, linewidth=0.5)

    ax4.set_title('Full Waveform Overview', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.legend(loc='upper right')
    fig4.tight_layout()
    plots['overview'] = fig_to_base64(fig4)
    plt.close(fig4)

    return plots


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')