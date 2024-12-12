import supfun_sequences as sf
import time
import sounddevice as sd
import numpy as np

sd.default.samplerate = 44100
sd.default.device = 2
def generate_sound_data(frequency, volume=10, waveform="sine", duration=15, fs=44100):
    """Generate sound data for a given frequency and waveform."""
    t = np.linspace(0, duration, int(fs * duration), False)
    sound = np.zeros_like(t)

    if waveform == 'sine':
        sound = np.sin(frequency * t * 2 * np.pi)

    return sound * volume

sound1 = generate_sound_data(10000)
sound2= generate_sound_data(14222)
sound3= generate_sound_data(16666)

sounds = [sound1, sound2, sound3]

for i in range(len(sounds)):
    sf.play_sound(sounds[i])
    time.sleep(10)
    if i<= 2:
        sf.play_interval(sounds[i], sounds[i+1])
        time.sleep(10)
    else:
        sf.play_interval(sounds[i], sounds[0])

    
print("done")