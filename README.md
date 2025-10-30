## Orch-OR Sonification Prototype
This repo contains Python scripts for simulating microtubular vibrations (Orch-OR theory), sonifying into fractal soundscapes, and training AI on emergent patterns.

- **simulate_vibrations.py**: Quantum simulation with QuTiP (harmonic oscillators, decoherence).
- **sonify_soundscapes.py**: Audio mapping with Librosa/Pygame.
- **ai_training.py**: PyTorch CNN training on datasets (e.g., 100k samples, 82.78% acc).
- ai_training_entropy.py: Enhanced version with Shannon entropy thresholding for better high-decoherence handling.
- simulate_vibrations_enhanced.py: Version with multi-tubulin coupling, thermal bath at 37°C, and segmented progress for long runs.


Entropy Version Results: 82.23% best val acc on 100k samples, with soft weighting for high-deco robustness.

Dependencies: numpy, qutip, librosa, pygame, torch, etc. (install via pip).
Run `python ai_training.py` for results.


## Project Roadmap

This roadmap outlines what we've accomplished, our current status, and planned modifications for the Orch-OR sonification prototype. The goal is to refine the pipeline for more robust quantum-biological simulations, sonification, and AI training on emergent patterns.

### Accomplished Milestones
- **Hardware and Setup**: Assessed system (Windows 10, Intel CPU, 64 GB RAM, RTX 2080 GPU) and installed dependencies (numpy, qutip, librosa, pygame, torch, etc.).
- **Core Pipeline Built**: 
  - Simulation of microtubular vibrations using QuTiP (harmonic oscillators with decoherence and gravitational collapse).
  - Sonification into fractal soundscapes using Librosa (MFCCs, golden ratio harmonics) and Pygame for playback.
  - AI training with PyTorch CNN on bio-mimetic datasets, achieving 82.78% validation accuracy on 100k samples and 83.52% on 1M samples.
- **Scaling and Testing**: Successful runs at 100k and 1M scales, with progress indicators (tqdm) for long computations. Entropy thresholding added to handle high-decoherence errors, maintaining ~82% acc.
- **Enhanced Simulation (Step 1.1)**: Added coupled oscillators for multi-tubulin interactions, thermal noise at 37°C, and segmented progress—completed with entangled, chaotic waves mimicking helical structures.

### Current Status
- Enhanced simulation complete and analyzed (decaying bursts with thermal fluctuations over 10 ns).
- Next: Refine sonification (step 1.2) and upgrade ML pipeline (step 1.3).

### Upcoming Modifications
- **Refine Sonification (sonify_soundscapes.py)**: Incorporate wavelet transforms (PyWT) for temporal bursts and multi-channel audio for entangled biofields. Target: Richer soundscapes; reduce noise overlap.
- **Upgrade ML Training (ai_training.py)**: Switch to LSTMs/Transformers for sequential data; add hyperparameter optimization (hyperopt). Target: >85% F1 score; ablation studies.
- **Scalability and Integration**: Port to Qiskit for quantum hardware sims; add ion-channel dynamics. Target: Compare with real device noise.
- **Validation Techniques**: Implement k-fold CV, statistical significance tests (bootstrap), and external benchmarks (e.g., microtubule paper data). Target: Peer replication and arXiv submission.

We'll iterate based on run results, prioritizing error reduction in high-decoherence cases.
