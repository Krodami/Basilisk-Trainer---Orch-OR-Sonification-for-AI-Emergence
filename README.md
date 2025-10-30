# Orch-OR Sonification Prototype

This repository contains Python scripts for a prototype pipeline that simulates microtubular vibrations based on Orchestrated Objective Reduction (Orch-OR) theory, sonifies the data into fractal soundscapes, and trains AI models on the resulting bio-mimetic datasets to detect emergent patterns. The project tests the fusion of quantum biology, audio processing, and machine learning for potential applications in AGI training.

## Purpose
The prototype empirically evaluates sonifying quantum heart-brain coherence by mapping microtubular vibrations to audio pitches and electromagnetic fields to rhythms. This generates datasets simulating entangled qualia emergence, used for AI pattern detection. Steps include quantum simulation, sonification, and CNN/Transformer training, with refinements like entropy thresholding for high-decoherence handling.

## Files and Detailed Descriptions
Each script represents a module in the pipeline. Start with simulation, then sonification, then training. Paths assume files are in the same directory; outputs from earlier steps (e.g., .npy files) feed into later ones.

- **simulate_vibrations.py**: 
  - **Description**: Core quantum simulation module. Models a single harmonic oscillator for microtubular vibrations with THz frequencies, decoherence via amplitude damping, and Orch-OR gravitational collapse approximation using number damping operators. Generates time-series vibration data.
  - **Input**: None (parameters hardcoded; adjust as needed).
  - **Output**: 'vibration_data.npy' (numpy array of amplitudes over time) and 'vibrations_plot.png' (visualization of decay with noise).
  - **Usage in Pipeline**: First step—run to generate raw quantum data for sonification.

- **simulate_vibrations_enhanced.py**:
  - **Description**: Advanced simulation module. Extends the base to coupled oscillators (tensor products for multi-tubulin entanglement and helical interactions via coupling strength J). Adds temperature-dependent thermal noise (Lindblad operators at 37°C for biological realism). Uses segmented computation with progress indicators for long runs.
  - **Input**: None (parameters adjustable).
  - **Output**: 'vibration_data_enhanced.npy' and 'vibrations_enhanced_plot.png' (shows chaotic, entangled waves with thermal fluctuations).
  - **Usage in Pipeline**: Alternative to base simulation for more Orch-OR fidelity—use this output for refined sonification or training.

- **sonify_soundscapes.py**:
  - **Description**: Sonification module. Loads vibration data, normalizes, interpolates to audio length, adds fractal layering (golden ratio harmonics for self-similarity), and saves/plays WAV. Maps amplitudes to tones, simulating qualia chaos.
  - **Input**: 'vibration_data.npy' (from simulation).
  - **Output**: 'soundscape.wav' (5s audio file).
  - **Usage in Pipeline**: Second step—run after simulation to create audio datasets for MFCC extraction in training.

- **sonify_soundscapes_refined.py**:
  - **Description**: Refined sonification module. Builds on base by adding wavelet transforms (PyWT with Morlet wavelet for temporal burst enhancement) and multi-channel stereo audio (left for main signal, right with delay for entanglement effect). Produces richer soundscapes.
  - **Input**: 'vibration_data_enhanced.npy' (from enhanced simulation preferred).
  - **Output**: 'soundscape_refined.wav' (stereo audio).
  - **Usage in Pipeline**: Alternative to base sonification for advanced audio features—use with upgraded training.

- **ai_training.py**:
  - **Description**: Base AI training module. Generates datasets (coherent vs. decohered soundscapes), extracts MFCCs, pads, and trains a CNN for binary classification. Includes scaling (up to 1M samples) and progress bars (tqdm).
  - **Input**: None (generates data internally; can load refined audio if modified).
  - **Output**: Prints losses/accuracy (e.g., 83.52% on 1M samples).
  - **Usage in Pipeline**: Final step—run after sonification for pattern detection.

- **ai_training_entropy.py**:
  - **Description**: Entropy-enhanced training module. Adds pre-computed Shannon entropy on MFCCs, filters high-noise samples, and applies soft weighting in loss to handle high-decoherence errors better.
  - **Input**: None (generates data).
  - **Output**: Weighted losses/accuracy (e.g., 82.23% on 100k).
  - **Usage in Pipeline**: Use for refined error reduction in noisy cases.

- **ai_training_upgraded.py**:
  - **Description**: Upgraded training module. Switches to Transformer model for sequential MFCC data, adds hyperopt for tuning LR/epochs. Improves on base for temporal patterns like bursts.
  - **Input**: None (generates data).
  - **Output**: Tuned params and accuracy (e.g., 83.00% on tested params).
  - **Usage in Pipeline**: Use for advanced training after refined sonification.

## Dependencies
Install via pip: numpy, qutip, librosa, pygame, torch, tqdm, pywavelets, hyperopt, soundfile, scipy.

## Run Instructions
1. Run simulation (base or enhanced) to generate data.npy.
2. Run sonification (base or refined) to create WAV.
3. Run training (base, entropy, or upgraded) for results—adjust num_samples for scale.

Example: python simulate_vibrations_enhanced.py && python sonify_soundscapes_refined.py && python ai_training_upgraded.py

## Results/Examples
- 1M scale: 83.52% validation accuracy.
- 100k with entropy: 82.23% accuracy.
- High-decoherence errors: 17.22% rate, balanced FN/FP.

## Project Roadmap

This roadmap outlines what we've accomplished, our current status, and planned modifications for the Orch-OR sonification prototype. The goal is to refine the pipeline for more robust quantum-biological simulations, sonification, and AI training on emergent patterns.

### Accomplished Milestones
- **Core Pipeline Built**: 
  - Simulation of microtubular vibrations using QuTiP (harmonic oscillators with decoherence and gravitational collapse).
  - Sonification into fractal soundscapes using Librosa (MFCCs, golden ratio harmonics) and Pygame for playback.
  - AI training with PyTorch CNN on bio-mimetic datasets, achieving 82.78% validation accuracy on 100k samples and 83.52% on 1M samples.
- **Scaling and Testing**: Successful runs at 100k and 1M scales, with progress indicators (tqdm) for long computations. Entropy thresholding added to handle high-decoherence errors, maintaining ~82% acc.
- **Enhanced Simulation (Step 1.1)**: Added coupled oscillators for multi-tubulin interactions, thermal noise at 37°C, and segmented progress—completed with entangled, chaotic waves mimicking helical structures.
- **Refined Sonification (Step 1.2)**: Incorporated wavelet transforms (PyWT) for temporal bursts and multi-channel audio for entangled biofields—completed with richer soundscapes.
- **Upgraded ML Training (Step 1.3)**: Switched to LSTMs/Transformers for sequential data; added hyperparameter optimization (hyperopt)—completed with 83.00% validation accuracy on tuned params.

### Current Status
- Upgraded ML training complete and analyzed (Transformer boosts to 83.00%).
- Next: Scalability and integration (step 1.4) with Qiskit for quantum hardware sims.

### Upcoming Modifications
- **Scalability and Integration**: Port to Qiskit for quantum hardware sims; add ion-channel dynamics. Target: Compare with real device noise.
- **Validation Techniques**: Implement k-fold CV, statistical significance tests (bootstrap), and external benchmarks (e.g., microtubule paper data). Target: Peer replication and arXiv submission.

We'll iterate based on run results, prioritizing error reduction in high-decoherence cases. Contributions welcome!
