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
from lpc import lpc

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
WINDOW_SIZE = 2 * FRAME_SIZE
NUM_DERIV_FEATS = 6
NUM_PITCH_FEATS = 7

# not sure where this number comes from but its 16ms at 48khz
# also could be 80% of the window size
PITCH_MAX_PERIOD = 768
# not sure where this number comes from but its 1.25ms at 48khz
# also could be 6.25% of window size
# or 1/4 of the 5ms that the banding is based off of
PITCH_MIN_PERIOD = 60
PITCH_BUF_SIZE = PITCH_MAX_PERIOD + WINDOW_SIZE
PITCH_FILTER_ORDER = 4


def compute_band_energy(stft_frame):
    '''Takes in stft (freq_size, 2) and outputs the per band
    energy of the stft as defined in RNNoise paper equation 2
    '''
    band_energies = np.zeros(NUM_BANDS, dtype=np.float32)

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
    #print(peak)
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
        ceps_hist_1_deque.append(np.zeros(NUM_BANDS, dtype=np.float32))
        ceps_hist_2_deque.append(np.zeros(NUM_BANDS, dtype=np.float32))
    features = np.zeros((inputs.shape[0], 35), dtype=np.float32)
    band_energies = np.zeros((inputs.shape[0], NUM_BANDS), dtype=np.float32)
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

def update_pitch_history(pitch_history, signal_in):
    # copy history without first frame in history
    pitch_history[:PITCH_BUF_SIZE - FRAME_SIZE] = pitch_history[FRAME_SIZE:]
    pitch_history[PITCH_BUF_SIZE - FRAME_SIZE:] = signal_in


def downsample_pitch_history(pitch_history):
    pitch_buf = np.zeros(pitch_history.shape[0] // 2, dtype=np.float32)
    for i in range(1, pitch_buf.shape[0]):
        pitch_buf[i] = 0.5 * ( (0.5 * (pitch_history[2*i - 1] + pitch_history[2*i + 1])) + pitch_history[2*i])
    pitch_buf[0] = 0.5 * ( 0.5 * pitch_history[1] + pitch_history[0])
    return pitch_buf


#def filter_pitch(pitch_buf, autocorr):


def rnnoise_style_xcorr(a, b, max_pitch, length):
    xcorr = np.zeros(max_pitch, dtype=np.float32)
    for i in range(max_pitch):
        xcorr[i] = a[:length].dot(b[i:length+i])
    return xcorr


def gaussian_lag_window(signal):
    '''Exponential decay lag windowing
    '''
    n = signal.shape[0]
    w = [1 - ((0.008 * i) ** 2) for i in range(n)]
    for i in range(n):
        signal[i] *= w[i]


def compute_autocorr_and_lag_for_lpc(pitch_buf, order):
    acorr = rnnoise_style_xcorr(pitch_buf, pitch_buf, order + 1, pitch_buf.shape[0] - order)
    for k in range(order + 1):
        d = 0
        for i in range(pitch_buf.shape[0] - order + k, pitch_buf.shape[0]):
            d += pitch_buf[i] * pitch_buf[i - k]
        acorr[k] += d
    # rnnoise code says that this enforces a noise floor of -40dB
    acorr[0] *= 1.0001

    gaussian_lag_window(acorr)
    return acorr

def find_best_pitch(xcorr, pitch_buf, max_pitch, length):
    '''Returns the first and second best pitchs based on an argmax over k
    of xcorr[k]**2 * ( 1 + \sum_i^n (y_i)^2 - \sum_i^k (y_i)^2 + \sum_{i=len}^{k+len} (y_i)^2)
    Not quite this because for each k we take a max of the second term, call it Syy, and 1
    There is another discrepancy because we accept a new best index,k, iff
    xcorr[k]^2 * best_syy > best_square_xcorr * current_syy
    haven't quite figured out what this^ is about yet
    '''
    #TODO(james): Come up with a more mathematical understanding of what this is doing
    two_best_indices = [0, 1]
    two_best_Syy = [0, 0]
    two_best_square_xcorr = [-1, -1]
    Syy = 1. + np.sum(np.square(pitch_buf[:length]))

    for i in range(max_pitch):
        if xcorr[i] > 0:
            xcorr_sq = xcorr[i] ** 2
            # check to see if this index is better than the second best one so far
            if xcorr_sq * two_best_Syy[1] > two_best_square_xcorr[1] * Syy:
                # if it is check to see if it is better than the best index
                if xcorr_sq * two_best_Syy[0] > two_best_square_xcorr[0] * Syy:
                    #best becomes second best
                    two_best_indices[1] = two_best_indices[0]
                    two_best_Syy[1] = two_best_Syy[0]
                    two_best_square_xcorr[1]  = two_best_square_xcorr[0]

                    two_best_indices[0] = i
                    two_best_Syy[0] = Syy
                    two_best_square_xcorr[0] = xcorr_sq
                else:
                    two_best_indices[1] = i
                    two_best_Syy[1] = Syy
                    two_best_square_xcorr[1] = xcorr_sq

        Syy += pitch_buf[i + length] ** 2
        Syy -= pitch_buf[i] ** 2
        Syy = max(1., Syy)
    return two_best_indices


def coarse_search(pitch_buf, max_pitch, delay):
    # use the first WINDOW_SIZE part of pitch_buf
    # note that its divided by 4 because it has been downsampled by
    # 2 twice now, also note the addition of max_pitch / 4, this is
    # since the xcorr will search across all indices up to max_pitch / 4
    # so the non-delayed version needs to be an extra max_pitch / 4 long
    length = WINDOW_SIZE // 4
    ds_pitch_buf = np.zeros(length + max_pitch // 4, dtype=np.float32)
    ds_pitch_buf_delayed = np.zeros(length, dtype=np.float32)
    # Downsample by 2x again
    for i in range(ds_pitch_buf.shape[0]):
        ds_pitch_buf[i] = pitch_buf[2*i]
        if i < ds_pitch_buf_delayed.shape[0]:
            ds_pitch_buf_delayed[i] = pitch_buf[2*i + delay]

    xcorr = rnnoise_style_xcorr(
            ds_pitch_buf_delayed,
            ds_pitch_buf,
            max_pitch // 4,
            ds_pitch_buf_delayed.shape[0]
    )
    two_best_pitches = find_best_pitch(xcorr, ds_pitch_buf, max_pitch // 4, length)
    return two_best_pitches

def fine_search(pitch_buf, max_pitch, two_best_pitches, delay):
    '''Finer search with only 2x downsampling
    '''
    xcorr = np.zeros(max_pitch // 2, dtype=np.float32)
    length = WINDOW_SIZE // 2
    pitch_buf_delayed = pitch_buf[delay:delay+length]
    for i in range(xcorr.shape[0]):
        # only search in a region that's within 2 indices either way of the best
        # two pitches
        if abs(i - 2 * two_best_pitches[0]) > 2 and abs(i - 2 * two_best_pitches[1]) > 2:
            continue
        xcorr[i] = pitch_buf_delayed.dot(pitch_buf[i:i+length])
        # don't ask me why we do this clipping
        xcorr[i] = max(-1., xcorr[i])

    two_best_pitches = find_best_pitch(xcorr, pitch_buf, max_pitch // 2, length)
    # pseudo interpolation of pitch indices
    if two_best_pitches[0] > 0 and two_best_pitches[0] < (max_pitch // 2 - 1):
        a = xcorr[two_best_pitches[0] - 1]
        b = xcorr[two_best_pitches[0]]
        c = xcorr[two_best_pitches[0] + 1]
        # heuristic for interpolating
        if (c - a) > 0.7 * (b - a):
            offset = 1
        elif (a - c) > 0.7 * (b - a):
            offset = -1
        else:
            offset = 0
    else:
        offset = 0
    return 2*two_best_pitches[0] - offset

def pitch_search(pitch_buf):
    max_pitch = PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD
    delay = PITCH_MAX_PERIOD // 2
    # coarse search first
    two_best_pitches = coarse_search(pitch_buf, max_pitch, delay)
    # then do fine search to find index
    pitch_index = fine_search(pitch_buf, max_pitch, two_best_pitches, delay)
    return pitch_index

def compute_pitch_features(signal, stft_frames):
    # stores signal history of size PITCH_BUF_SIZE
    pitch_history = np.zeros((PITCH_BUF_SIZE,), dtype=np.float32)
    num_frames = signal.shape[0] // FRAME_SIZE
    pitch_features = np.zeros((num_frames, NUM_PITCH_FEATS), dtype=np.float32)

    for i in range(num_frames):
        frame = signal[i * FRAME_SIZE: (i + 1) * FRAME_SIZE]
        update_pitch_history(pitch_history, frame)
        #pitch_buf is downsampled version of pitch history of length PITCH_BUF_SIZE / 2
        pitch_buf = downsample_pitch_history(pitch_history)
        acorr = compute_autocorr_and_lag_for_lpc(pitch_buf, PITCH_FILTER_ORDER)
        lpc_coeff = lpc(acorr, PITCH_FILTER_ORDER)
        # FIR filter based on lpc coefficients
        filtered_buf = scipy.signal.lfilter(lpc_coeff, [1.0], pitch_buf)
        pitch_index = pitch_search(filtered_buf)
        #print(pitch_index)





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
    #for i in range(signal.shape[1] // 480):
    #    pitch_freq = freq_from_autocorr(signal[0, i*480:(i+1)*480], 48000)
    #    print(pitch_freq)
    flat_signal = (pcm_float_to_i16(signal.flatten()) / 10).astype(np.float32) / 960.
    stft_settings = VorbisWindowSTFTSettings(sample_rate=48000, window_length=0.02, hop_length=0.01)
    stft_frames = stft(flat_signal, stft_settings)
    stft_frames_r = np.real(stft_frames[0])
    stft_frames_i = np.imag(stft_frames[0])
    stft_frames = np.stack([stft_frames_r, stft_frames_i], axis=-1)
    stft_frames = stft_frames[:1314]
    compute_pitch_features(flat_signal * 960, stft_frames)

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
