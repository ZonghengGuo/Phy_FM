import numpy as np
import vitaldb

import matplotlib.pyplot as plt


path = "preprocessing/0001.vital"
vf = vitaldb.VitalFile(path)

ECG_ii = vf.to_numpy(['SNUADC/ECG_II'], 1 / 100)
ECG_v = vf.to_numpy(['SNUADC/ECG_V5'], 1 / 100)
ECG_ppg = vf.to_numpy(['SNUADC/PLETH'], 1 / 100)

print(vf)

# vf_ECG_ii = vitaldb.read_vital(path, track_names)
# ECG_ii = vf_ECG_ii.to_numpy(track_names, 1 / 100)

time_axis = np.arange(0, 10, 0.01)
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs[0].plot(time_axis, ECG_ii[20000:21000], color='blue')
axs[0].set_title('ECG Lead II (SNUADC/ECG_II)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

axs[1].plot(time_axis, ECG_v[20000:21000], color='green')
axs[1].set_title('ECG Lead V (SNUADC/ECG_V)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

axs[2].plot(time_axis, ECG_ppg[20000:21000], color='red')
axs[2].set_title('Photoplethysmogram (SNUADC/PLETH)')
axs[2].set_ylabel('Amplitude')
axs[2].set_xlabel('Time (seconds)')

plt.tight_layout()

plt.show()