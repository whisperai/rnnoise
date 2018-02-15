from __future__ import print_function
import argparse
from collections import deque
import os

import h5py
import scipy
from scipy import fftpack
from scipy.signal import fftconvolve
from matplotlib.mlab import find
import numpy as np

import whisper.utils as utils
from whisper.signal.stft import stft
from whisper.signal.datatype import pcm_float_to_i16, pcm_i16_to_float
from whisper.viz.stft import plot_magnitude
from custom_stft_window import VorbisWindowSTFTSettings

BAND_FREQS = np.array([
    0, 200, 400, 600, 800, 1000,
    1200, 1400, 1600, 2000, 2400,
    2800, 3200, 4000, 4800, 5600,
    6800, 8000, 9600, 12000, 15600,
    20000,
])

FRAME_LENGTH = 10e-3
MULTIPLES_OF_2_5ms = FRAME_LENGTH / 2.5e-3
# This is an array that looks like
# [0., 4., 8., 12., 16., 20., 24., 28., ...]
# where each element is the index of the frequency of the start of that band
# I DONT UNDERSTAND WHY ITS 2 * FRAME_LENGTH * BAND_FREQS instead of
# FRAME_LENGTH * BAND_FREQS
#SAMPLE_INDEX_OF_BAND = (BAND_FREQS * 5e-3 * MULTIPLES_OF_2_5ms).astype(np.int32)
SAMPLE_INDEX_OF_BAND = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])
#TODO(james): move this into a function since its not a constant
NUM_BANDS = BAND_FREQS.size
CEPS_MEM_SIZE = 8
FRAME_SIZE = 480
NUM_DERIV_FEATS = 6

NUM_DERIV_FEATS

def compute_band_energy(stft_frame):
    '''Takes in stft (freq_size, 2) and outputs the per band
    energy of the stft as defined in RNNoise paper equation 2
    '''
    band_energies = np.zeros(NUM_BANDS)

    for i in range(NUM_BANDS - 1):
        band_size = (SAMPLE_INDEX_OF_BAND[i + 1] - SAMPLE_INDEX_OF_BAND[i]) << 2
        for j in range(int(band_size)):
            freq_energy = np.sum(np.square(stft_frame[(SAMPLE_INDEX_OF_BAND[i]<< 2) + j]))
            # response to get smooth banding that has a triangle waveforms that look
            # like: |
            #       | |
            #       |   |
            #       |     |
            #       |       |
            # for the current band and:
            #               |
            #             | |
            #           |   |
            #         |     |
            #       |       |
            # for the next band
            triangle_resp_1 = float(j) / band_size
            triangle_resp_0 = 1 - triangle_resp_1
            band_energies[i] += triangle_resp_0 * freq_energy
            band_energies[i + 1] += triangle_resp_1 * freq_energy

    # multiply first and last band energies by 2 to account for the edge cases
    # of the triangular bands
    band_energies[0] *= 2
    band_energies[-1] *= 2
    return band_energies

def compute_log_energies(band_energies):
    #TODO(james): figure out what all these constants do
    # if I know what the constants do theres probably a fast
    # numpy way to do the for loop
    log_energies = np.log10(band_energies + 0.01)
    # running max energy
    log_max = -2
    follow = -2
    for i in range(log_energies.shape[0]):
        log_energies[i] = max(log_max - 7, max(follow - 1.5, log_energies[i]))
        log_max = max(log_max, log_energies[i])
        follow = max(follow - 1.5, log_energies[i])
    return log_energies


def dct(bands):
    '''Takes in a sequence of length NUM_BANDS and
    returns the DCT coefficients of that sequence
    '''
    return scipy.fftpack.dct(bands, norm='ortho')

def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, currently has trouble with finding the true peak

    """
    # Calculate circular autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]

    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    # Also could zero-pad before doing circular autocorrelation.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def get_MFCC_and_derivatives(
        stft_frame,
        cepstral_history_1_deque,
        cepstral_history_2_deque,
    ):
    '''Takes in stft_frame of dimension: (freq_size, 2), and outputs 22 MFCC coefficients,
    and the first and second time derivative of the first 6 MFCCs,
    for a total of 34 features
    '''
    # concatenate on the time axis
    band_energies = compute_band_energy(stft_frame)
    log_energies = compute_log_energies(band_energies)
    E = np.sum(band_energies)
    mfcc = dct(log_energies)
    #TODO(james): figure out what these constants represent
    mfcc[0] -= 12
    mfcc[1] -= 4

    # last set of MFCCs
    prev_cepstrum = cepstral_history_1_deque[-1]
    cepstral_history_1_deque.append(mfcc)
    # set before the last set of MFCCs
    prev_prev_cepstrum = cepstral_history_2_deque[-1]
    cepstral_history_2_deque.append(prev_cepstrum)

    first_n_mfccs = mfcc[:NUM_DERIV_FEATS] + prev_cepstrum[:NUM_DERIV_FEATS] + \
            prev_prev_cepstrum[:NUM_DERIV_FEATS]
    # this is an order O(delta_t **2) approximation instead of the typical
    # x_t - x_t-1 order O(delta_t) approximation since we have to keep x_t-2 around for
    # second derivative anyway
    first_n_deltas = mfcc[:NUM_DERIV_FEATS] - prev_prev_cepstrum[:NUM_DERIV_FEATS]
    first_n_ddeltas = mfcc[:NUM_DERIV_FEATS] - 2 * prev_cepstrum[:NUM_DERIV_FEATS] + \
            prev_prev_cepstrum[:NUM_DERIV_FEATS]
    return np.concatenate([first_n_mfccs, mfcc[NUM_DERIV_FEATS:], first_n_deltas, first_n_ddeltas]), band_energies


def get_spectral_variability(cepstral_history_deque):
    '''Takes in a history of the last CEPS_MEM_SIZE frames' MFCCs
    outputs magic spectral variability
    '''
    history_length = len(cepstral_history_deque)
    spec_variability = 0
    cepstral_history_arr = np.stack([cepstrum for cepstrum in cepstral_history_deque])
    for i in range(history_length):
        min_dist = 1e15
        for j in range(history_length):
            dist_sum = 0
            for k in range(NUM_BANDS):
                pairwise_dist = cepstral_history_arr[i, k] - cepstral_history_arr[j, k]
                dist_sum += pairwise_dist ** 2
            # ignore dist of 0 when i = j
            if j != i:
                min_dist = min(min_dist, dist_sum)
        spec_variability += min_dist
    spec_variability /= CEPS_MEM_SIZE
    # Don't know what this constant means
    spec_variability -= 2.1
    return spec_variability


def get_features_from_stfts(inputs):
    # history of cepstrums
    ceps_hist_1_deque = deque(maxlen=CEPS_MEM_SIZE)
    # history of cepstrums delayed by 2
    ceps_hist_2_deque = deque(maxlen=CEPS_MEM_SIZE)
    for i in range(CEPS_MEM_SIZE):
        ceps_hist_1_deque.append(np.zeros(NUM_BANDS))
        ceps_hist_2_deque.append(np.zeros(NUM_BANDS))
    features = np.zeros((inputs.shape[0], 35))
    band_energies = np.zeros((inputs.shape[0], NUM_BANDS))
    for i in range(inputs.shape[0]):
        mfccs, band_e = get_MFCC_and_derivatives(
                inputs[i],
                ceps_hist_1_deque,
                ceps_hist_2_deque,
        )
        #for x in inputs[i]:
        #    print("{},{};  ".format(x[0], x[1]), end="")
        #print()
        spec_var = get_spectral_variability(ceps_hist_1_deque)
        features[i] = np.hstack([mfccs, spec_var])
        band_energies[i] = band_e

    return features, band_energies

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_h5')
    parser.add_argument('feats_h5')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print('Loading data...')
    with h5py.File(args.feats_h5, 'r') as hf:
        feats = hf['data'][:]
    with h5py.File(args.input_h5, 'r') as hf:
        inputs = hf['data'][:]
    with h5py.File('extra_data', 'r') as hf:
        input_sequences = hf['data'][:]
    print('done.')

    signal, _ = utils.load_wav(os.path.expanduser("test_wavs/speech/dir0-head0-sample_31c15490b50150b5cc6957a6235fa2524be74d19.json.wav"))
    flat_signal = pcm_float_to_i16(signal.flatten()).astype(np.float32) / 960.
    stft_settings = VorbisWindowSTFTSettings(sample_rate=48000, window_length=0.02, hop_length=0.01)
    stft_frames = stft(flat_signal, stft_settings)
    stft_frames_r = np.real(stft_frames[0])
    stft_frames_i = np.imag(stft_frames[0])
    stft_frames = np.stack([stft_frames_r, stft_frames_i], axis=-1)
    stft_frames = stft_frames[:1314]


    print("Generating features in python")
    feats_py, energies = get_features_from_stfts(inputs)
    #feats_py, energies = get_features_from_stfts(stft_frames)
    print('done.')


    with h5py.File('py_feats.h5', 'w') as hf:
        hf.create_dataset('data', data=feats_py)
    with h5py.File('energies.h5', 'w') as hf:
        hf.create_dataset('data', data=energies)
    mse = np.mean(np.square(feats_py[:,-1] - feats[:, -1]))
    print("MSE: {}".format(mse))
    print(feats_py[:, -1])
    print(feats[:, -1])
    #print("MSE energy: {}".format(mse_energy))
