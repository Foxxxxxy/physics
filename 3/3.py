import numpy as np
from matplotlib import pyplot as plot
from scipy.fft import irfft, rfft, rfftfreq

R = 40000
D = 10
F = 500
NF = 5000

PLOT_LENGTH = 1000


def generate_signal(frequency, rate, duration):
    x = np.linspace(0, duration, rate * duration, endpoint=False)
    fr = x * frequency
    return np.sin(2 * np.pi * fr)


# ORIGINAL SIGNAL
y = generate_signal(F, R, D)
plot.plot(y[:PLOT_LENGTH])
plot.show()

# ORIGINAL SIGNAL SPECTRUM
y_fourier = rfft(y)
x_fourier = rfftfreq(R * D, 1 / R)

plot.plot(x_fourier, np.abs(y_fourier))
plot.show()

# ADDING NOISE
noise_t = generate_signal(NF, R, D) * 0.25
out = y + noise_t

plot.plot(out[:PLOT_LENGTH])
plot.show()

# SIGNAL WITH NOISE SPECTRUM
y_fourier = rfft(out)
x_fourier = rfftfreq(R * D, 1 / R)

plot.plot(x_fourier, np.abs(y_fourier))
plot.show()

# DELETE NOISE
points = len(x_fourier) / (R / 2)

target_idx = int(points * NF)
y_fourier[target_idx - 5:target_idx + 5] = 0

plot.plot(x_fourier, np.abs(y_fourier))
plot.show()

# INVERSE FOURIER TRANSFORM
new_signal = irfft(y_fourier)

plot.plot(new_signal[:PLOT_LENGTH])
plot.show()

# SAME ACTIONS WITH THREE HARMONIC SIGNALS
t1 = generate_signal(F, R, D)
t2 = generate_signal(F * 5, R, D)
t3 = generate_signal(F * 50, R, D)
y = t1 + t2 + t3

plot.plot(y[:PLOT_LENGTH])
plot.show()

y_fourier = rfft(y)
xf = rfftfreq(R * D, 1 / R)

plot.plot(xf, np.abs(y_fourier))
plot.show()

noise_t = generate_signal(NF, R, D)
noise_t = noise_t * 0.25
out = y + noise_t

plot.plot(out[:1000])
plot.show()

y_fourier = rfft(out)
xf = rfftfreq(R * D, 1 / R)

plot.plot(xf, np.abs(y_fourier))
plot.show()

points = len(xf) / (R / 2)

target_idx = int(points * NF)
y_fourier[target_idx - 5:target_idx + 5] = 0

plot.plot(xf, np.abs(y_fourier))
plot.show()

new_signal = irfft(y_fourier)

plot.plot(new_signal[:1000])
plot.show()
