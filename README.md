## Orch-OR Sonification Prototype
This repo contains Python scripts for simulating microtubular vibrations (Orch-OR theory), sonifying into fractal soundscapes, and training AI on emergent patterns.

- **simulate_vibrations.py**: Quantum simulation with QuTiP (harmonic oscillators, decoherence).
- **sonify_soundscapes.py**: Audio mapping with Librosa/Pygame.
- **ai_training.py**: PyTorch CNN training on datasets (e.g., 100k samples, 82.78% acc).
- ai_training_entropy.py: Enhanced version with Shannon entropy thresholding for better high-decoherence handling.

Entropy Version Results: 82.23% best val acc on 100k samples, with soft weighting for high-deco robustness.

Dependencies: numpy, qutip, librosa, pygame, torch, etc. (install via pip).
Run `python ai_training.py` for results.
