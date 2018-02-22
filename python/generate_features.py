'''Example usage:

python python/generate_features.py --noise_dir ~/code/whisper/runpppy/noises/ --speech_dir /Users/emmett/data/eval_wav/skynest-main-room-2017-12-21/voices/tt/
'''
from __future__ import print_function
import argparse
import subprocess
import glob
import os
import random

import h5py
import numpy as np

import whisper.utils as utils
from whisper.signal.datatype import pcm_float_to_i16

# FRAME_SIZE from RNNoise
FRAME_SIZE = 480
FREQ_SIZE = FRAME_SIZE + 1
SAMPLE_RATE = 48000

DENOISE_TRAINING_BIN = './bin/denoise_training'

def iter_wavs(dir_path, sample_rate):
    path_list = glob.glob(os.path.join(dir_path, '*.wav'))
    for path in path_list:
        yield utils.load_wav(path, sample_rate)[0]

def wav_dir_to_binary_file(dir_path, out_path, flatten_channels):
    '''Load all wavs in directory and output to a binary file.'''
    wavs = iter_wavs(dir_path, SAMPLE_RATE)

    # Add each channel's data to output array.
    # NOTE(emmett): Flattening channels may not be useful, and may cause over weighting of some
    # data captured with Shannon vs other mics.
    # However the additional diversity should help with real world performance, ensuring the model
    # learns general bits and not things specific to certain microphones.
    if flatten_channels:
        max_channels = 1024
    else:
        max_channels = 1

    signal_parts = []
    for wav in wavs:
        # TODO(emmett): Maybe add augmentation here?
        for c in range(min(max_channels, wav.shape[0])):
            signal_parts.append(wav[c, :])
        # TODO(emmett): if we only add one channel we could stream out to disk rather than store
        # whole array in memory

    #random.shuffle(signal_parts)
    maga_wav = np.concatenate(signal_parts)
    i16 = pcm_float_to_i16(maga_wav) / 10
    i16.tofile(out_path)

    return i16

def wav_to_binary_file(wav_path, out_path):
    signal, _ = utils.load_wav(wav_path, sr=SAMPLE_RATE)
    arr = pcm_float_to_i16(signal)
    with open(out_path, 'wb') as f:
        arr.tofile(f)
    return arr

def run_rnnoise_feature_binary(speech_path, noise_path, out_path, num_frames):
    args = [
        DENOISE_TRAINING_BIN,
        speech_path,
        noise_path,
        str(num_frames),
        #str(1),
    ]
    with open(out_path, 'wb') as f:
        subprocess.check_call(args, stdout=f)

    # 87 float32 features
    # plus a (FREQ_SIZE, 2) stft
    dt = np.dtype("({},)f4".format(2*FREQ_SIZE + 480 + 42))
    stdout_arr = np.fromfile(out_path, dtype=dt)
    feature_arr = stdout_arr[:, 2*FREQ_SIZE+480:]
    input_arr = stdout_arr[:, :2 * FREQ_SIZE]
    extra_arr = stdout_arr[:, 2*FREQ_SIZE:2*FREQ_SIZE+480]
    input_arr = input_arr.reshape((-1, 2, FREQ_SIZE))
    input_arr = np.transpose(input_arr, (0, 2, 1))
    return feature_arr, input_arr, extra_arr

def build_training_bin():
    print('Building training binary...')
    subprocess.check_call('mkdir -p bin', shell=True)
    command = 'cd src; gcc -DTRAINING=1 -Wall -W -O3 -g -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o ../bin/denoise_training -lm'
    subprocess.check_call(command, shell=True)
    print('Done!')

def get_features(speech_wav_dir, noise_wav_dir):
    if not os.path.exists(DENOISE_TRAINING_BIN):
        build_training_bin()

    assert os.path.exists(DENOISE_TRAINING_BIN), 'Missing compiled denoise bin!'

    speech_out_path = '/tmp/speech.bin'
    noise_out_path = '/tmp/noise.bin'
    features_out_path = '/tmp/feats.bin'

    print('Creating speech mega wav...')
    speech = wav_dir_to_binary_file(speech_wav_dir, speech_out_path, flatten_channels=True)
    print('Creating noise mega wav...')
    noise = wav_dir_to_binary_file(noise_wav_dir, noise_out_path, flatten_channels=True)

    assert len(speech.shape) == 1, 'Only mono audio supported not {}'.format(speech.shape)
    assert len(noise.shape) == 1, 'Only mono audio supported not {}'.format(noise.shape)

    num_frames = speech.shape[0] / FRAME_SIZE
    print("Generating features for {} frames".format(num_frames))
    feature_arr, input_arr, extra_arr = run_rnnoise_feature_binary(
        speech_out_path,
        noise_out_path,
        features_out_path,
        num_frames,
    )
    return feature_arr, input_arr, extra_arr

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_dir', required=True)
    parser.add_argument('--noise_dir', required=True)
    parser.add_argument('h5_out', help='Output h5 dataset for trainer')
    return parser.parse_args()

def main():
    args = get_args()
    feat_arr, input_arr, extra_arr = get_features(args.speech_dir, args.noise_dir)
    print(extra_arr.shape)

    print('Creating h5 dataset...')
    h5f = h5py.File(args.h5_out, 'w')
    h5f.create_dataset('data', data=feat_arr)
    h5f.close()
    h5f_input = h5py.File('stft_input.h5', 'w')
    h5f_input.create_dataset('data', data=input_arr)
    h5f_input.close()
    h5f_extra = h5py.File('extra_data', 'w')
    h5f_extra.create_dataset('data', data=extra_arr)
    h5f_extra.close()
    print('Done!')


if __name__ == '__main__':
    main()
