import numpy as np
import librosa
import pygame
from pygame.mixer import Sound
import soundfile as sf
import pywt  # For wavelet transforms

# Load the enhanced vibration data (from step 1.1)
full_data = np.load('vibration_data_enhanced.npy')  # Use your new file

# Normalize data
data_norm = full_data / np.max(np.abs(full_data))

# Sonification parameters
sample_rate = 44100
duration = 5.0
num_samples = int(sample_rate * duration)
time_stretch = np.linspace(0, 1, num_samples)

from scipy.interpolate import interp1d
interpolator = interp1d(np.linspace(0, 1, len(data_norm)), data_norm, kind='cubic')
audio_data = interpolator(time_stretch)

# Add wavelet transform for temporal bursts (using Morlet wavelet)
print("Milestone: Applying wavelet transform...")
coeffs = pywt.cwt(audio_data, scales=np.arange(1, 129), wavelet='morl')
wavelet_audio = coeffs[0].mean(axis=0) * 0.2  # Average scales for burst enhancement

# Fractal layering (golden ratios)
fractal_ratios = [1, 1.618, 2.618]
fractal_audio = np.zeros_like(audio_data)
for ratio in fractal_ratios:
    overtone = np.sin(2 * np.pi * ratio * 440 * time_stretch) * audio_data * 0.3
    fractal_audio += overtone

# Multi-channel for entangled biofields (stereo: left for main vibrations, right for shifted/entangled version)
soundscape_left = audio_data + fractal_audio + wavelet_audio
soundscape_right = np.roll(soundscape_left, int(sample_rate * 0.05))  # 50ms delay for entanglement effect
soundscape_stereo = np.vstack((soundscape_left, soundscape_right)).T
soundscape_stereo = soundscape_stereo / np.max(np.abs(soundscape_stereo))

# Save and play
sf.write('soundscape_refined.wav', soundscape_stereo, sample_rate)
print("Refined soundscape generated and saved as 'soundscape_refined.wav'.")

pygame.mixer.init()
sound = Sound('soundscape_refined.wav')
sound.play()
print("Playing sound... (5 seconds)")
pygame.time.wait(int(duration * 1000))
pygame.mixer.quit()
print("Milestone: Sonification complete!")