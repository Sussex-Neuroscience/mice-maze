import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


#if using a focusrite soundcard, refer to this https://support.focusrite.com/hc/en-gb/articles/115004120965-Sample-Rate-Bit-Depth-Buffer-Size-Explained

sd.default.samplerate = 192000 #available sample rates: 44.1kHz, 48kHz , 88.2kHz, 96kHz, 176.4kHz , 192 kHz
sd.default.device = 2 # Replace this by setting found_channel = False in  "__main__"

  
#this function generates an array of sound data according to the duration, frequency, volume and sample rate. 
# If the sample rate fs is not defined, the default is 44100 Hz
def generate_sound_data(frequency, volume=1, waveform="sine", duration=15, fs=44100):
    """Generate sound data for a given frequency and waveform."""
    t = np.linspace(0, duration, int(fs * duration), False)
    sound = np.zeros_like(t)

    if waveform == 'sine':
        sound = np.sin(frequency * t * 2 * np.pi)

    return sound * volume

#This function normalises the sound array and plays the sound using the sd.play. Sounds can also be played non normalised 

def play_sound(sound_data, fs=44100):
    """Play sound using sounddevice library."""
    sound_data_normalised = np.int16((sound_data / np.max(np.abs(sound_data))) * 32767)
    sd.play(sound_data_normalised, fs)

#this function plays a series of sounds. It is used to check that the actual output of the code (frequency and volume of the sound) are correct
#use this and modify it to calibrate volume and sample rate
def check_sounds(sample_rate):
    sound1 = generate_sound_data(10000, volume = 0.5, fs= sample_rate)
    sound2= generate_sound_data(14222, volume = 0.5, fs= sample_rate)
    sound3= generate_sound_data(16666,volume = 0.5, fs= sample_rate)
    sound4= generate_sound_data(22000,volume = 0.1, fs=sample_rate)
    sound5= generate_sound_data(32000,volume = 10, fs=sample_rate)
    sound6= generate_sound_data(40000,volume = 10, fs=sample_rate)


    sounds = [sound1, sound2, sound3, sound4, sound5, sound6]

    

    for i in range(len(sounds)):
        print(f"playing sound{i}")
        sd.play(sounds[i], sample_rate)

        time.sleep(10)
    #     # if i<= 2:
        #     sf.play_interval(sounds[i], sounds[i+1])
        #     time.sleep(10)
        # else:
        #     sf.play_interval(sounds[i], sounds[0])

        
    print("done")
    

#you can use this function to visualise the fft of the generated sound.

def make_FFT(sound, sample_rate):

    # Compute the FFT
    fft_result = np.fft.fft(sound)
    fft_freq = np.fft.fftfreq(len(sound), 1/sample_rate)

    # Plot the FFT
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    plt.title('FFT of the Sound')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()



#find which channels are used to send the signal to the sondcard. The channel will have to be set as sd.default.device


def find_channel(sound):
    #this function runs through 50 channels to find the ones connected to the soundcard. 
    #50 is an arbitrary number, to find the precise number of channels, run 'python -m sounddevice' or 'python3 -m sounddevice' in the terminal
    for i in range(50):
        sd.default.device=i
        try: 
            play_sound(sound)
            print(f"using device {i}")
            time.sleep(5)
        except:
            print(f"unable to play with device {i}")
            pass


if __name__== "__main__":

    trial_sound = generate_sound_data(10000, volume = 1, fs= 44100)
    
    #set to false when using for the first time
    found_channel= False
   
    if not found_channel:
        # this will generate a list of channels that can or cannot play the sound. Many channels will return the same output. 
        #some channels will distort the sound. Once you find the channel, go to the top of the code and change sd.default.device. 
        # Then set found_channel = True
        find_channel(trial_sound)


        #change the sample rate to calibrate the sounds
        sample_rate = 192000
        check_sounds(sample_rate)

        #set to true if you want to visualise the FFT of the sound
        visualise_FFT = False
        if visualise_FFT:
            make_FFT(trial_sound, sample_rate)




        










