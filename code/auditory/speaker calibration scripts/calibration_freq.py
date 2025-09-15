import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import time
import pandas as pd
from scipy.interpolate import interp1d



#This is 
sd.default.samplerate = 192000 #available sample rates: 44.1kHz, 48kHz , 88.2kHz, 96kHz, 176.4kHz , 192 kHz
sd.default.device = (None, 4)


def compute_gain(frequency_hz):


    data = pd.read_csv("C:/Users/labuser/Documents/GitHub/mice-maze/code/auditory/speaker calibration scripts/frequency_response_speaker.csv")
    # interpolation function from the frequency response data extrapolated from the graph on the website
    frequency = data['Frequency_kHz'].values * 1000  # Convert kHz to Hz
    attenuation = data['Attenuation_dB'].values
    interp_func = interp1d(frequency, attenuation, kind='cubic', fill_value="extrapolate")
    #get the interpolated attenuation
    attenuation_db = interp_func(frequency_hz)
    #from there calculate the gain 
    gain = 10 ** (-attenuation_db / 20)
    return gain


    

 

def generate_sound_data(frequency, waveform="sine", duration=10, fs=192000, volume=1.0, ramp_duration = 0.01, compensate=True, give_t= False):
    """Generate sound data for a given frequency and waveform."""
    t = np.linspace(0, duration, int(fs * duration), endpoint = False)
    gain = compute_gain(frequency) if compensate else 1.0
    adjusted_volume = volume * gain
    sound = np.zeros_like(t)    


    if waveform == 'sine':
        sound = np.sin(frequency * t * 2 * np.pi)  * adjusted_volume

    elif waveform == 'square':
        sound = np.sign(np.sin(frequency * t * 2 * np.pi)) * adjusted_volume

    elif waveform == 'sawtooth':
        sound = (2 * (t * frequency % 1) - 1) * adjusted_volume

    elif waveform == 'triangle':
        sound = (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1) * adjusted_volume

    elif waveform == 'pulse wave':
        sound = np.where((t % (1 / frequency)) < (1 / frequency) * 0.5, 1.0, -1.0)  * adjusted_volume  # 0.5 in this case is the duty cycle
    
    elif waveform == 'white noise':
        samples = int(fs * duration)
        sound = np.random.uniform(low=-1.0, high=1.0, size=samples) * adjusted_volume

    
    # apply a ramp up (fade in) in seconds so that the speaker doesn't click upon stimulus presentation
    ramp_samples = int(fs * ramp_duration)

    envelope = np.ones_like(sound)
    if ramp_samples > 0:
        # Linear ramp up
        envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
        # Linear ramp down
        # envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples)

    sound *= envelope

    if give_t:
        return t, sound
    else:
        return sound


def play_sound(sound_data, fs=192000, volume=1.0):
    """Play sound using sounddevice library with improved volume scaling."""
    # sound_data = np.clip(sound_data, -1, 1)  # Ensure within valid range
    # sound_data_scaled = np.int16(sound_data * 32767 * volume)  # Apply volume scaling in dB range
    # sd.play(sound_data_scaled, fs)
    # sd.wait()
    sd.play(sound_data, fs)
    sd.wait()


def start_calibration():
    try:
        sample_rate = int(sample_rate_entry.get())
        num_sounds = int(num_sounds_entry.get())
        sounds = []

        for i in range(num_sounds):
            freq = float(freq_entries[i].get())*1000 #convert Hz to KHz
            vol = float(vol_entries[i].get())
            sound = generate_sound_data(freq, volume=vol, fs=sample_rate)
            sounds.append((sound,vol))

        messagebox.showinfo("Info", "Playing sounds for calibration...")
        
        for i, (sound,vol) in enumerate(sounds):
            play_sound(sound, sample_rate, volume=vol)
            time.sleep(1)  # Pause between sounds
        
        messagebox.showinfo("Info", "Calibration Complete!")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

def update_sound_inputs():
    for widget in sound_frame.winfo_children():
        widget.destroy()
    
    global freq_entries, vol_entries
    freq_entries = []
    vol_entries = []
    num_sounds = int(num_sounds_entry.get())

    for i in range(num_sounds):
        tk.Label(sound_frame, text=f"Sound {i+1} Frequency (KHz):").grid(row=i, column=0)
        freq_entry = tk.Entry(sound_frame)
        freq_entry.grid(row=i, column=1)
        freq_entries.append(freq_entry)

        tk.Label(sound_frame, text=f"Sound {i+1} Volume (0-1):").grid(row=i, column=2)
        vol_entry = tk.Entry(sound_frame)
        vol_entry.grid(row=i, column=3)
        vol_entry.insert(0,"1")
        vol_entries.append(vol_entry)

# Create main UI window
root = tk.Tk()
root.title("Speaker Calibration Tool")
root.geometry("500x400")

# Sample rate input
tk.Label(root, text="Sample Rate (Hz):").pack()
sample_rate_entry = tk.Entry(root)
sample_rate_entry.pack()
sample_rate_entry.insert(0, "192000")  # Default value

# Number of sounds input
tk.Label(root, text="Number of Sounds:").pack()
num_sounds_entry = tk.Entry(root)
num_sounds_entry.pack()
num_sounds_entry.insert(0, "3")  # Default value

# Button to update sound inputs
tk.Button(root, text="Set Number of Sounds", command=update_sound_inputs).pack()

# Frame for frequency and volume inputs
sound_frame = tk.Frame(root)
sound_frame.pack()
update_sound_inputs()

# Start Calibration Button
tk.Button(root, text="Start Calibration", command=start_calibration).pack()

# Run the UI
root.mainloop()
