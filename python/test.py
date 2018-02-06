import numpy as np
from whisper.signal.datatype import pcm_i16_to_float
from whisper.utils import save_wav

dt = np.dtype(np.int16)
arr = np.fromfile('out.bin', dtype=dt)
farr = pcm_i16_to_float(arr)
save_wav('out.wav', farr, sr=48000)
