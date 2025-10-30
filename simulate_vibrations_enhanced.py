import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 15
omega = 2 * np.pi * 1e12
kappa = 1e8
grav_rate = 1e8
full_tlist = np.linspace(0, 1e-8, 1000)

print("Milestone: Setting up operators...")
a1 = qt.tensor(qt.destroy(N), qt.identity(N))
a2 = qt.tensor(qt.identity(N), qt.destroy(N))
J = 1e11
H = omega * (a1.dag() * a1 + a2.dag() * a2) + J * (a1.dag() * a2 + a1 * a2.dag())

psi0 = qt.tensor(qt.coherent(N, 3.0), qt.coherent(N, 2.0))

T = 310
kB = 1.38e-23
hbar = 1.05e-34
n_th = 1 / (np.exp(hbar * omega / (kB * T)) - 1)
c_ops = [
    np.sqrt(kappa * (n_th + 1)) * a1, np.sqrt(kappa * n_th) * a1.dag(),
    np.sqrt(kappa * (n_th + 1)) * a2, np.sqrt(kappa * n_th) * a2.dag(),
    np.sqrt(grav_rate) * (a1.dag() * a1 + a2.dag() * a2)
]

options = {'nsteps': 50000, 'atol': 1e-8, 'rtol': 1e-6, 'progress_bar': True}  # Increased nsteps

print("Milestone: Starting segmented simulation...")
num_segments = 10
segment_size = len(full_tlist) // num_segments
states = [psi0]
vibrations = []

current_state = psi0
for seg in tqdm(range(num_segments), desc="Simulation segments"):
    print(f"Starting segment {seg+1}/{num_segments}...")  # Per-segment feedback
    start_idx = seg * segment_size
    end_idx = (seg + 1) * segment_size if seg < num_segments - 1 else len(full_tlist)
    tlist_seg = full_tlist[start_idx:end_idx]

    result_seg = qt.mesolve(H, current_state, tlist_seg, c_ops=c_ops, options=options)
    states.extend(result_seg.states[1:])
    current_state = result_seg.states[-1]

    x = a1 + a1.dag() + a2 + a2.dag()
    vibrations_seg = qt.expect(x, result_seg.states)
    vibrations.extend(vibrations_seg)
    print(f"Segment {seg+1} complete!")

print("Milestone: Simulation complete!")

vibrations = np.array(vibrations)

fractal_noise = np.cumsum(np.random.randn(len(full_tlist))) * 0.05
biofield = np.sin(2 * np.pi * 40 * full_tlist) * 0.3

full_data = vibrations + fractal_noise + biofield

plt.figure(figsize=(10, 5))
plt.plot(full_tlist * 1e9, full_data)
plt.xlabel('Time (ns)')
plt.ylabel('Vibration Amplitude')
plt.title('Enhanced Coupled Orch-OR Simulation with Segmented Progress')
plt.savefig('vibrations_enhanced_plot.png')
plt.show()

np.save('vibration_data_enhanced.npy', full_data)
print("Enhanced coupled simulation complete with segmented progress!")