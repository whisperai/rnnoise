from __future__ import print_function
import argparse
import subprocess

import numpy as np

import whisper.utils as utils
from whisper.signal.datatype import pcm_float_to_i16

# FRAME_SIZE from RNNoise
FRAME_SIZE = 480

def wav_to_binary_file(wav_path, out_path):
    signal, _ = utils.load_wav(wav_path)
    arr = pcm_float_to_i16(signal)
    with open(out_path, 'wb') as f:
        arr.tofile(f)
    return arr

def run_rnnoise_feature_binary(speech_path, noise_path, out_path, num_frames):
    args = [
        "./bin/denoise_training",
        speech_path,
        noise_path,
        str(num_frames),
    ]
    with open(out_path, 'wb') as f:
        subprocess.check_call(args, stdout=f)

    # 87 float32 features
    dt = np.dtype("(87,)f4")
    feature_arr = np.fromfile(out_path, dtype=dt)
    return feature_arr


def get_features(speech_wav_path, noise_wav_path):
    speech_out_path = '/tmp/speech.bin'
    noise_out_path = '/tmp/noise.bin'
    features_out_path = '/tmp/feats.bin'

    speech = wav_to_binary_file(speech_wav_path, speech_out_path)
    noise = wav_to_binary_file(noise_wav_path, noise_out_path)

    num_frames = speech.shape[1] / FRAME_SIZE
    print("Generating features for {} frames".format(num_frames))
    feature_arr = run_rnnoise_feature_binary(
        speech_out_path,
        noise_out_path,
        features_out_path,
        num_frames,
    )
    return feature_arr

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_path', required=True)
    parser.add_argument('--noise_path', required=True)
    return parser.parse_args()

def main():
    args = get_args()
    feat_arr = get_features(args.speech_path, args.noise_path)
    print(feat_arr.shape)

if __name__ == '__main__':
    main()
