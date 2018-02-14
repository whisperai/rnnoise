import scipy
import numpy as np

from whisper.signal.stft import STFTSettings

class VorbisWindowSTFTSettings(STFTSettings):

    def get_window(self):
        window_samples = int(self.sample_rate * self.window_length)
        n = np.arange(window_samples)
        w = np.sin((np.pi / 2) * np.square(np.sin((np.pi * (n + 0.5)) / window_samples)))
        return w

