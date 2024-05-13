import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq


SAMPLE_RATE = 30  # Hertz
DURATION = 10  # Seconds

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y1 = generate_sine_wave(1, SAMPLE_RATE, DURATION)
x, y2 = generate_sine_wave(5, SAMPLE_RATE, DURATION)

y = y1+ y2
plt.figure()
plt.plot(x, y)
plt.show()

N = SAMPLE_RATE * DURATION

yf = fft(y)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.figure()
plt.plot(xf, np.abs(yf))
plt.show()