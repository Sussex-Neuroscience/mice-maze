import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import get_window
import tkinter as tk
from tkinter import filedialog as fd

def choose_csv():
    root=tk.Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename


INPUT_FILE = choose_csv()

#set SPEED_UP to True to speed up to rate to set to speed_rate
SPEED_UP= True
SPEED_RATE = 3

if not SPEED_UP: 
    SPEED_RATE = 1



# Load the .wav file
sample_rate, data = wavfile.read(INPUT_FILE)

# If stereo, take only one channel
if data.ndim > 1:
    data = data[:, 0]

# Parameters
window_size = 2048  # Number of samples per FFT
hop_size = 512     # Number of samples between successive FFTs
window = get_window('hann', window_size)
duration = len(data) / sample_rate  # Duration of the audio in seconds
time = np.arange(0, len(data)) / sample_rate  # Time axis for the audio signal

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Static Spectrogram
ax1.specgram(data, NFFT=window_size, Fs=sample_rate, noverlap=window_size - hop_size, cmap='viridis')
ax1.set_title('Spectrogram')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

# Real-time FFT
line, = ax2.plot([], [], lw=2)
ax2.set_xlim(0, 45000)
ax2.set_ylim(0, 1)
ax2.set_title('Real-time FFT')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Amplitude')

# Add a text box for the current time
time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)

# Initialize the FFT line
def init():
    line.set_data([], [])
    return line,

# Update the FFT line
def update(frame):
    start = frame * hop_size  * SPEED_RATE 
    end = start + window_size
    if end > len(data):
        return line,
    segment = data[start:end] * window
    spectrum = np.abs(fft(segment))[:window_size // 2]
    spectrum = spectrum / np.max(spectrum)  # Normalize
    freqs = np.linspace(0, sample_rate / 2, window_size // 2)
    line.set_data(freqs, spectrum)
    current_time = (start / sample_rate) 
    time_text.set_text(f'Time: {current_time:.2f}s')
    return line, time_text

# Create animation
frames = (len(data) - window_size) // (hop_size * SPEED_RATE )
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=(hop_size / sample_rate * 1000)/SPEED_RATE)

plt.tight_layout()
plt.show()