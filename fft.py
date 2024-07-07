import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 500
T = 1.0 / sampling_rate
t = np.linspace(0.0, 2.0, int(sampling_rate), endpoint=False)  # 시간 벡터

num_frequencies = 4 
frequencies = np.random.uniform(20, 200, num_frequencies)
amplitudes = np.random.uniform(0.1, 1.0, num_frequencies)

signal = np.zeros_like(t)
for i in range(num_frequencies):
    signal += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)

signal_fft = np.fft.fft(signal)
frequencies_fft = np.fft.fftfreq(len(signal), T)

filtered_signals = []
for target_freq in frequencies:
    filter_mask = np.zeros_like(signal_fft, dtype=bool)

    filter_mask[np.abs(frequencies_fft - target_freq) < 1] = True
    filtered_signal_fft = signal_fft * filter_mask
    filtered_signal = np.fft.ifft(filtered_signal_fft)
    filtered_signals.append(filtered_signal.real)

fig, axes = plt.subplots(num_frequencies + 1, 1, figsize=(10, 12))

axes[0].plot(t, signal)
axes[0].set_title("원본 주파수")
axes[0].set_xlabel("시간")
axes[0].set_ylabel("진폭")

# 필터링된 신호들
for i, filtered_signal in enumerate(filtered_signals):
    axes[i + 1].plot(t, filtered_signal)
    axes[i + 1].set_title(f"변형 {frequencies[i]:.2f} Hz")
    axes[i + 1].set_xlabel("시간")
    axes[i + 1].set_ylabel("진폭")

plt.tight_layout()
plt.show()
