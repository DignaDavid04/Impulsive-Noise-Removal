import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import warnings

# Suppress DeprecationWarning for scipy.io.wavfile.read
warnings.filterwarnings("ignore", category=DeprecationWarning)

def remove_impulsive_noise(input_filepath, output_filepath, ar_rank=4, eta=3.0, forgetting_factor=0.99):
    """
    Removes impulsive noise from a WAV file using an autoregressive model and linear interpolation.

    Args:
        input_filepath (str): Path to the input WAV file.
        output_filepath (str): Path to save the cleaned WAV file.
        ar_rank (int): The rank (order) of the AR model.
        eta (float): The threshold multiplier for noise detection (typically 1.5 to 4).
        forgetting_factor (float): The forgetting factor (lambda) for the EW-LS algorithm.
    """
    print("Reading audio file...")
    try:
        samplerate, data = wavfile.read(input_filepath)
        original_dtype = data.dtype
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return

    # --- 1. Pre-processing ---
    # Convert to mono by averaging channels if stereo
    if data.ndim > 1:
        audio = data.mean(axis=1)
    else:
        audio = data

    # Normalize to floating point between -1 and 1 for numerical stability
    # Use a small epsilon to avoid division by zero for silent files
    max_abs = np.max(np.abs(audio))
    if max_abs == 0:
        max_abs = 1.0
    audio_norm = audio / max_abs
    n_samples = len(audio_norm)
    cleaned_audio = np.copy(audio_norm)

    print("Initializing model parameters...")
    # --- 2. Initialization for EW-LS ---
    r = ar_rank
    lam = forgetting_factor
    
    # AR model coefficients
    a = np.zeros(r)
    
    # Covariance matrix P, initialized to a large value (identity * 1000)
    P = np.eye(r) * 1000
    
    # Initial estimate of the prediction error variance
    error_var_estimate = 0.01

    # Lists to store data for plotting
    prediction_errors = []
    detection_thresholds = []
    ar_coeffs_history = [[] for _ in range(r)]
    detected_clicks_indices = []

    print("Processing audio samples...")
    # --- 3. Main Processing Loop ---
    # Start after the first 'r' samples and leave enough samples at the end for processing
    k = r
    while k < n_samples - (r + 1):
        # Create the vector of past 'r' cleaned samples
        y_past = np.flip(cleaned_audio[k-r:k])

        # Predict the current sample based on the AR model
        y_pred = np.dot(a, y_past)
        
        # Calculate prediction error against the original (potentially noisy) sample
        error = audio_norm[k] - y_pred

        # Update the local estimate of error standard deviation (delta_e)
        error_var_estimate = lam * error_var_estimate + (1 - lam) * error**2
        delta_e = np.sqrt(error_var_estimate)
        threshold = eta * delta_e

        # Store for plotting
        prediction_errors.append(error)
        detection_thresholds.append(threshold)

        # --- 4. Noise Detection and Correction ---
        if abs(error) > threshold:
            detected_clicks_indices.append(k)
            # Impulsive noise detected. Find the duration of the noise burst (max 4 samples).
            click_start = k
            click_end = k
            
            # Check the next 3 samples to see if they are also part of the burst
            # We check up to r-1 samples ahead (so loop range(1, r))
            for j in range(1, r):
                if k + j >= n_samples - 1:
                    break
                
                # Temporarily use cleaned audio history to predict the next sample
                next_y_past = np.flip(cleaned_audio[k+j-r:k+j])
                next_y_pred = np.dot(a, next_y_past)
                next_error = audio_norm[k+j] - next_y_pred
                
                # Use the same threshold for consistency within the burst detection
                if abs(next_error) > threshold:
                    click_end = k + j
                else:
                    break
            
            # --- 5. Linear Interpolation for Noise Removal ---
            # Replace the bad samples with a linear interpolation between the last known
            # good sample before the burst and the first good sample after it.
            num_bad_samples = click_end - click_start + 1
            
            # Ensure we are not at the very edge of the file
            if click_start > 0 and click_end < n_samples - 1:
                y_start_good = cleaned_audio[click_start - 1]
                y_end_good = audio_norm[click_end + 1] # Use original audio for future sample
                
                for i in range(num_bad_samples):
                    idx_to_replace = click_start + i
                    # Calculate the weight for interpolation
                    alpha = (i + 1) / (num_bad_samples + 1)
                    cleaned_audio[idx_to_replace] = (1 - alpha) * y_start_good + alpha * y_end_good
            
            # Skip the main loop to the end of the corrected segment
            k = click_end + 1
            continue

        # --- 6. EW-LS Parameter Update (only for clean samples) ---
        # This update is only performed when a sample is deemed 'clean'.
        g_num = P @ y_past
        g_den = lam + y_past.T @ g_num
        g = g_num / g_den  # Kalman gain vector
        P = (1 / lam) * (P - np.outer(g, y_past) @ P) # Update covariance matrix
        
        a = a + g * error # Update AR coefficients

        for i in range(r):
            ar_coeffs_history[i].append(a[i])
        
        k += 1

    print("Post-processing and saving file...")
    # --- 7. Post-processing ---
    # De-normalize the audio back to its original range
    cleaned_audio_denorm = cleaned_audio * max_abs
    
    # Convert back to the original data type (e.g., int16)
    cleaned_audio_final = cleaned_audio_denorm.astype(original_dtype)

    wavfile.write(output_filepath, samplerate, cleaned_audio_final)
    print(f"Cleaned audio saved to {output_filepath}")

    # --- 8. Generate Plots for Report ---
    print("Generating analysis plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Find a good region to plot by locating a detected click
    if detected_clicks_indices:
        center_of_plot = detected_clicks_indices[len(detected_clicks_indices)//2]
        plot_range_start = max(0, center_of_plot - 1000)
        plot_range_end = min(n_samples, center_of_plot + 1000)
    else: # Default if no clicks found
        print("No impulsive noise was detected to center the plot. Using default range.")
        plot_range_start = max(0, n_samples // 2 - 1000)
        plot_range_end = min(n_samples, n_samples // 2 + 1000)

    time_axis = np.arange(n_samples) / samplerate
    
    # Plot 1 & 2: Waveform comparison and Prediction error
    fig1, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    axs[0].plot(time_axis[plot_range_start:plot_range_end], audio_norm[plot_range_start:plot_range_end], label='Original Noisy Signal', color='red', alpha=0.7)
    axs[0].plot(time_axis[plot_range_start:plot_range_end], cleaned_audio[plot_range_start:plot_range_end], label='Cleaned Signal', color='blue', alpha=0.7)
    axs[0].set_title('Waveform Comparison (Zoomed In)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    
    plot_indices = range(len(prediction_errors))
    # Adjust time axis for error plot, as it starts from sample 'r'
    time_axis_err = time_axis[r:r+len(plot_indices)]
    if len(time_axis_err) == len(prediction_errors):
        axs[1].plot(time_axis_err, prediction_errors, label='Prediction Error', color='gray', alpha=0.6)
        axs[1].plot(time_axis_err, detection_thresholds, label='Positive Threshold (η·δe)', color='darkorange', linestyle='--')
        axs[1].plot(time_axis_err, -np.array(detection_thresholds), label='Negative Threshold (-η·δe)', color='darkorange', linestyle='--')
    
    if detected_clicks_indices:
        # Filter clicks that are within the plotted error range
        valid_click_times = []
        valid_click_values = []
        for idx in detected_clicks_indices:
            time_val = idx / samplerate
            # Ensure the click is within the time axis of the error plot
            if idx - r >= 0 and idx - r < len(prediction_errors):
                valid_click_times.append(time_val)
                valid_click_values.append(prediction_errors[idx - r])

        axs[1].scatter(valid_click_times, valid_click_values, color='red', s=50, zorder=5, label='Detected Clicks')
    
    axs[1].set_title('Prediction Error and Adaptive Detection Threshold')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Error Amplitude')
    axs[1].legend()
    
    # Set x-limit for zoom
    if plot_range_start < plot_range_end:
        axs[1].set_xlim(time_axis[plot_range_start], time_axis[plot_range_end])
    
    fig1.tight_layout()
    fig1.savefig('report_figure_1_waveform_error.png')
    
    # Plot 3: AR Coefficients
    fig2, ax = plt.subplots(figsize=(15, 6))
    for i in range(r):
        if ar_coeffs_history[i]: # Check if not empty
            ax.plot(ar_coeffs_history[i], label=f'a{i+1}')
    ax.set_title('Evolution of AR Model Parameters (a_i)')
    ax.set_xlabel('Processing Step (Clean Samples)')
    ax.set_ylabel('Coefficient Value')
    ax.legend()
    fig2.tight_layout()
    fig2.savefig('report_figure_2_ar_coeffs.png')
    
    print("Plots saved as 'report_figure_1_waveform_error.png' and 'report_figure_2_ar_coeffs.png'")


if __name__ == '__main__':
    # Define file paths and parameters
    INPUT_FILE = '11.wav' 
    OUTPUT_FILE = '11_cleaned.wav'
    
    # Parameters based on the project description and common practice
    AR_ORDER = 4
    THRESHOLD_ETA = 3.5 # A value between 1.5 and 4. A higher value is more conservative.
    FORGETTING_FACTOR_LAMBDA = 0.99 # Controls how quickly the model adapts. Closer to 1 means slower adaptation.
    
    # Run the noise removal process
    remove_impulsive_noise(
        input_filepath=INPUT_FILE,
        output_filepath=OUTPUT_FILE,
        ar_rank=AR_ORDER,
        eta=THRESHOLD_ETA,
        forgetting_factor=FORGETTING_FACTOR_LAMBDA
    )