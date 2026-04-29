import numpy as np

fs = 48000
ws = 40
D = 8
M = 1024
Q = 4   # or 8

bitrate = (fs / ws / D) * Q * np.log2(M)

print("Bitrate:", bitrate/1000, "kbps")
