import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

def find_amplitude(input_wav, start_time, end_time, N, target_freq):
    sample_rate, data = wavfile.read(input_wav)
    if data.ndim > 1:
        data = data[:, 0]  # Use one channel if stereo

    # Define the time range where the tone is steady

    start_index = int(start_time * sample_rate)
    end_index = int(end_time * sample_rate)
    segment = data[start_index:end_index]

    # FFT parameters
    # FFT length
    hop_size = N // 2          # 50% overlap
    window = get_window('hann', N)
    window_cg = np.mean(window)  # Coherent gain (≈0.5 for Hann window)
    window = window/ window_cg

    # Create the frequency axis for one block
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)

    # Find the FFT bin closest to 16 kHz
    target_idx = np.argmin(np.abs(freqs - target_freq))

    # Initialize list to collect amplitude values
    amplitudes = []
    amplitudes_raw = []

    # Slide the window across the segment
    for start in range(0, len(segment) - N + 1, hop_size):
        block = segment[start:start + N]
        # Apply the Hann window to reduce spectral leakage
        windowed_block = block * window
        # Compute the FFT of the windowed block
        fft_block = np.fft.rfft(windowed_block)
        # Extract the raw amplitude at the target frequency
        amp_raw = np.abs(fft_block[target_idx])
        # Correct for FFT scaling and window attenuation
        amp = (2.0 / N) * (amp_raw)
        amplitudes.append(amp)
        amplitudes_raw.append(amp_raw)

    # Compute the average amplitude at the target frequency
    A_avg = np.mean(amplitudes)
    A_avg_raw = np.mean(amplitudes_raw)
    #print(f"Average amplitude at {target_freq}Hz: {A_avg} / raw:{A_avg_raw}")

    return A_avg, A_avg_raw

def find_calibration_constant(SPL, amplitude):
    C= SPL - 20*(np.log10(amplitude)) 
    return C


