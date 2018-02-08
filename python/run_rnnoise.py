'''Run inference with compiled RNNoise weights.'''
from __future__ import print_function

import argparse
import os
import subprocess
import tempfile

import numpy as np

from whisper.utils import load_wav, save_wav
from whisper.signal.datatype import pcm_float_to_i16
from whisper.signal.datatype import pcm_i16_to_float
from whisper.signal.stft import STFTSettings, stft, istft
    
from generate_features import SAMPLE_RATE

RNNOISE_DEMO_BIN = './examples/rnnoise_demo'

def run_rnnoise(input_wav):
    assert os.path.exists(RNNOISE_DEMO_BIN), 'Must compile rnnoise first!'

    with tempfile.NamedTemporaryFile() as temp_in, \
         tempfile.NamedTemporaryFile() as temp_out:
        pcm_float_to_i16(input_wav).tofile(temp_in)
        command = [RNNOISE_DEMO_BIN, temp_in.name, temp_out.name]
        subprocess.check_call(command)
        output_i16 = np.fromfile(temp_out.name, dtype=np.int16)
    output_f = pcm_i16_to_float(output_i16)
    return output_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('speech_wav')
    parser.add_argument('output_wav')
    parser.add_argument('noise_wav')
    args = parser.parse_args()

    speech_wav, _ = load_wav(args.speech_wav, sr=SAMPLE_RATE)
    speech_mono_wav = speech_wav[0, :]
    noise_wav, _ = load_wav(args.noise_wav, sr=SAMPLE_RATE)
    noise_mono_wav = noise_wav[0, :]

    combined_mono_wav = speech_mono_wav + noise_mono_wav[:speech_mono_wav.shape[0]]
    combined_mono_wav /= np.abs(combined_mono_wav).max()

    stft_settings = STFTSettings(sample_rate=SAMPLE_RATE)
    speech_stft = np.abs(stft(speech_mono_wav, stft_settings))
    noise_stft = np.abs(stft(noise_mono_wav[:speech_mono_wav.shape[0]], stft_settings))
    ibm = 1 - np.argmax(np.stack([speech_stft, noise_stft]), axis=0)

    save_wav('noisy_speech.wav', combined_mono_wav, sr=SAMPLE_RATE)
    output_mono_wav = run_rnnoise(combined_mono_wav)
    save_wav(args.output_wav, output_mono_wav, sr=SAMPLE_RATE)
    #uncomment and comment above two line to skip running rnnoise
    #output_mono_wav, _ = load_wav(args.output_wav, sr=SAMPLE_RATE)

    mse_in = np.mean((combined_mono_wav - speech_mono_wav)**2)
    mse = np.mean((output_mono_wav - speech_mono_wav[:output_mono_wav.shape[0]]) ** 2)
    print("MSE IN: {}, MSE OUT: {}".format(mse_in, mse))

    out_stft = stft(output_mono_wav, stft_settings)
    ibm_target = istft(out_stft * ibm[:,:out_stft.shape[1]], stft_settings)

    save_wav('ibm_target.wav', ibm_target, sr=SAMPLE_RATE)

if __name__ == '__main__':
    main()
