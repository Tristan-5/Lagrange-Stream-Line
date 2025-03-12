import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# --- Parameters for 2D Synthetic Turbulence ---
N = 256           # Number of grid points in each dimension
L = 10.0          # Domain size (assumed square)
dx = L / N        # Grid spacing

# Create coordinate grid (not explicitly needed for FFT, but useful)
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# --- Generate the Fourier grid ---
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)
K[0, 0] = 1e-6  # Avoid division by zero at k=0

# --- Prescribe a power-law spectrum for the streamfunction ---
# For Kolmogorov scaling in the inertial range, we set amplitude ~ k^(-5/3 - 1)
# The extra -1 arises because the energy in 2D comes from |psi|^2 and velocity ~ grad(psi)
exponent = -(5/3 + 1)
amplitude = K**exponent

# Add random phases
np.random.seed(42)
random_phase = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
psi_hat = amplitude * random_phase

# Ensure the field is real by imposing Hermitian symmetry:
psi_hat = fftshift(psi_hat)
psi_hat[N//2+1:, :] = np.conj(np.flipud(np.fliplr(psi_hat[1:N//2, :])))
psi_hat = np.fft.ifftshift(psi_hat)

# --- Inverse Fourier Transform to obtain the streamfunction in real space ---
psi = np.real(ifft2(psi_hat))

# --- Derive the velocity field from the streamfunction (2D incompressible field) ---
# u = d(psi)/dy, v = -d(psi)/dx
# Compute gradients using spectral differentiation
ikx = 1j * KX
iky = 1j * KY
psi_hat = fft2(psi)
u_hat = iky * psi_hat
v_hat = -ikx * psi_hat
u = np.real(ifft2(u_hat))
v = np.real(ifft2(v_hat))

# --- Compute the 2D Energy Spectrum ---
# The energy per mode is 0.5*(|u_hat|^2 + |v_hat|^2)
E_k2D = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

# Radially average the energy spectrum.
# Define bins for k
k_flat = K.flatten()
E_flat = E_k2D.flatten()
k_bins = np.linspace(0, np.max(K), 50)
k_bin_centers = 0.5*(k_bins[1:] + k_bins[:-1])
E_radial = np.zeros_like(k_bin_centers)

for i in range(len(k_bin_centers)):
    bin_inds = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
    if np.any(bin_inds):
        E_radial[i] = np.mean(E_flat[bin_inds])
    else:
        E_radial[i] = np.nan

# --- Plot the Energy Spectrum ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(psi, extent=[0, L, 0, L], origin='lower')
plt.colorbar(label='Streamfunction')
plt.title("Synthetic 2D Streamfunction")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.loglog(k_bin_centers, E_radial, 'o-', label='Synthetic 2D Spectrum')
# Plot reference -5/3 slope: we set an arbitrary constant for comparison
ref = k_bin_centers**(-5/3)
# Normalize reference line to overlay on the data for visual comparison
ref = ref * (E_radial[~np.isnan(E_radial)][0] / ref[0])
plt.loglog(k_bin_centers, ref, 'k--', label=r'Reference $k^{-5/3}$')
plt.xlabel("Wavenumber k")
plt.ylabel("Energy Spectrum E(k)")
plt.title("Radially Averaged Energy Spectrum")
plt.legend()
plt.tight_layout()
plt.show()

# --- Optional: Compare with Real Simulation Data ---
# If you have a 2D velocity field from a DNS or experiment (e.g., stored as numpy arrays u_real, v_real),
# you can compute the Fourier transform and perform the same radial averaging.
#
# For example:
# u_real = np.load('u_real.npy')
# v_real = np.load('v_real.npy')
# u_real_hat = fft2(u_real)
# v_real_hat = fft2(v_real)
# E_real = 0.5*(np.abs(u_real_hat)**2 + np.abs(v_real_hat)**2)
# Then perform a similar radial average as done above.
#
# This allows you to overlay the real data's spectrum with the synthetic one and compare the scaling.

print("Script complete. Check the plots for comparison with the Kolmogorov -5/3 scaling.")
