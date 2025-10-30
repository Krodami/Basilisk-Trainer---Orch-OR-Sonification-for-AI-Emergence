import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

N = 30  # Energy levels
omega = 2 * np.pi * 1e12  # Frequency
kappa = 1e8  # Adjusted decoherence for stability
grav_rate = 1e8  # Adjusted gravity collapse rate
tlist = np.linspace(0, 1e-8, 2000)  # Time array

a = qt.destroy(N)
H = omega * (a.dag() * a)

psi0 = qt.coherent(N, 3.0)

# Collapse operators
c_ops = [np.sqrt(kappa) * a, np.sqrt(grav_rate) * a.dag() * a]

# Add Options for integrator
options = qt.Options(nsteps=25000, atol=1e-8, rtol=1e-6)  # Increased nsteps; tolerances for precision

result = qt.mesolve(H, psi0, tlist, c_ops=c_ops, options=options)  # Pass options

x = a + a.dag()
vibrations = qt.expect(x, result.states)

fractal_noise = np.cumsum(np.random.randn(len(tlist))) * 0.05
biofield = np.sin(2 * np.pi * 40 * tlist) * 0.3

full_data = vibrations + fractal_noise + biofield

plt.figure(figsize=(10, 5))
plt.plot(tlist * 1e9, full_data)
plt.xlabel('Time (ns)')
plt.ylabel('Vibration Amplitude')
plt.title('Enhanced Orch-OR Simulation (Stable)')
plt.savefig('vibrations_plot.png')
plt.show()

np.save('vibration_data.npy', full_data)
print("Enhanced simulation complete!")