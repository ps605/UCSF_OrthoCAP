import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

# Filtering            
f_order = 2
f_cutoff = np.array([1,5])
f_sampling = 1000
f_nyquist = f_cutoff/(f_sampling/2)
b, a = signal.butter(f_order, f_nyquist, btype='band', analog = False)

SAMPLE_RATE = 1000  # Hertz
DURATION = 10  # Seconds

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y1 = generate_sine_wave(1, SAMPLE_RATE, DURATION)
x, y2 = generate_sine_wave(3, SAMPLE_RATE, DURATION)
x, y3 = generate_sine_wave(5, SAMPLE_RATE, DURATION)

y = y1 + y2 + y3
yfilt = signal.filtfilt(b, a, y , axis=0)

plt.figure()
plt.plot(x, y, c='r')
plt.plot(x, yfilt, c='g')

plt.show()

N = SAMPLE_RATE * DURATION

yf = fft(y)
yffilt = fft(yfilt)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.figure()
plt.plot(xf, np.abs(yf), c='r')
plt.plot(xf, np.abs(yffilt), c='g')
plt.show()