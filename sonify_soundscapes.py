import numpy as np
import librosa
import pygame
import soundfile as sf  # New import for writing WAV
from pygame.mixer import Sound, get_init

# Load the simulated vibration data
full_data = np.load('vibration_data.npy')

# Normalize data to audio range (e.g., -1 to 1 for WAV)
data_norm = full_data / np.max(np.abs(full_data))  # Normalize amplitudes

# Sonification parameters
sample_rate = 44100  # Standard audio sample rate (Hz)
duration = 5.0  # Stretch to 5 seconds for audible playback (slow down quantum timescale)
num_samples = int(sample_rate * duration)
time_stretch = np.linspace(0, 1, num_samples)  # Resample time

# Interpolate data to audio length
from scipy.interpolate import interp1d
interpolator = interp1d(np.linspace(0, 1, len(data_norm)), data_norm, kind='cubic')
audio_data = interpolator(time_stretch)

# Add fractal layering: Generate self-similar overtones (e.g., harmonics at fractal ratios)
fractal_ratios = [1, 1.618, 2.618]  # Golden ratio for fractal feel (self-similar)
fractal_audio = np.zeros_like(audio_data)
for ratio in fractal_ratios:
    overtone = np.sin(2 * np.pi * ratio * 440 * time_stretch) * audio_data * 0.3  # 440 Hz base pitch modulated
    fractal_audio += overtone

# Combine and normalize final soundscape
soundscape = audio_data + fractal_audio
soundscape = soundscape / np.max(np.abs(soundscape))  # Re-normalize

# Save as WAV file using soundfile
sf.write('soundscape.wav', soundscape, sample_rate)
print("Soundscape generated and saved as 'soundscape.wav'.")

# Playback with Pygame (simple audio player)
pygame.mixer.init()
sound = Sound('soundscape.wav')
sound.play()
print("Playing sound... (5 seconds)")
pygame.time.wait(int(duration * 1000))  # Wait for playback to finish
pygame.mixer.quit()