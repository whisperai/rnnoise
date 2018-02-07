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
    parser.add_argument('input_wav')
    parser.add_argument('output_wav')
    args = parser.parse_args()

    input_wav, _ = load_wav(args.input_wav, sr=SAMPLE_RATE)
    input_mono_wav = input_wav[0, :]
    output_mono_wav = run_rnnoise(input_mono_wav)
    save_wav(args.output_wav, output_mono_wav, sr=SAMPLE_RATE)


if __name__ == '__main__':
    main()
