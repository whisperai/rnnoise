from __future__ import print_function
import argparse
from collections import deque
import os
import sys

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
NUM_FEATS = 1 + NUM_PITCH_FEATS + NUM_BANDS + 2 * NUM_DERIV_FEATS

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

def compute_band_corr(stft_frame, pitch_stft_frame):
    '''Takes in a stft and a pitch delayed stft and computes the per band
    correlation of the two
    See compute_band_energy for more documentation
    '''
    band_corrs = np.zeros(NUM_BANDS, dtype=np.float32)
    for i in range(NUM_BANDS - 1):
        band_size = (SAMPLE_INDEX_OF_BAND[i + 1] - SAMPLE_INDEX_OF_BAND[i]) << 2

        for j in range(int(band_size)):
            current_X = stft_frame[(SAMPLE_INDEX_OF_BAND[i]<<2) + j]
            current_P = pitch_stft_frame[(SAMPLE_INDEX_OF_BAND[i]<<2) + j]
            # sums real and imaginary components
            freq_corr = np.sum(current_X * current_P)

            triangle_resp_1 = float(j) / band_size
            triangle_resp_0 = 1 - triangle_resp_1
            band_corrs[i] += triangle_resp_0 * freq_corr
            band_corrs[i + 1] += triangle_resp_1 * freq_corr

    band_corrs[0] *= 2
    band_corrs[-1] *= 2
    return band_corrs

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


def get_features(stfts, signal):
    # stores signal history of size PITCH_BUF_SIZE
    pitch_history = np.zeros((PITCH_BUF_SIZE,), dtype=np.float32)
    prev_pitch_index = 0
    prev_pitch_gain = 0
    # history of cepstrums
    ceps_hist_1_deque = deque(maxlen=CEPS_MEM_SIZE)
    # history of cepstrums delayed by 2
    ceps_hist_2_deque = deque(maxlen=CEPS_MEM_SIZE)
    for i in range(CEPS_MEM_SIZE):
        ceps_hist_1_deque.append(np.zeros(NUM_BANDS, dtype=np.float32))
        ceps_hist_2_deque.append(np.zeros(NUM_BANDS, dtype=np.float32))
    features = np.zeros((stfts.shape[0], NUM_FEATS), dtype=np.float32)
    band_energies = np.zeros((stfts.shape[0], NUM_BANDS), dtype=np.float32)
    for i in range(stfts.shape[0]):
        mfccs, band_e = get_MFCC_and_derivatives(
                stfts[i],
                ceps_hist_1_deque,
                ceps_hist_2_deque,
        )
        pitch_features, prev_pitch_index, prev_pitch_gain = compute_pitch_feats_for_frame(
                stfts[i],
                signal[i * FRAME_SIZE: (i + 1) * FRAME_SIZE],
                band_e,
                pitch_history,
                prev_pitch_index,
                prev_pitch_gain,
        )
        spec_var = get_spectral_variability(ceps_hist_1_deque)

        features[i] = np.hstack([mfccs, pitch_features, spec_var])
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
        elif (a - c) > 0.7 * (b - c):
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

def compute_pitch_gain(xx, yy, xy):
    return xy / np.sqrt(1 + xx * yy)

def remove_pitch_doubling(
        pitch_buf,
        current_pitch_ind,
        prev_pitch_period,
        prev_pitch_gain
    ):
    # account for using downsampled pitch buf
    min_period = PITCH_MIN_PERIOD // 2
    max_period = PITCH_MAX_PERIOD // 2
    prev_pitch_period /= 2
    prev_pitch_period = max_period - prev_pitch_period
    length = WINDOW_SIZE // 2
    T0 = max_period - (current_pitch_ind / 2)
    best_T = T0

    # from now on xx represents autocorr of pitch_buf delayed by max_period
    # xy represents xcorr of pitch_buf delayed by max_period with pitch_buf delayed by current_pitch_ind
    # yy_lookup[i] represents autocorr of pitch_buf delayed by i

    yy_lookup = np.zeros(max_period + 1, dtype=np.float32)
    x = pitch_buf[max_period:max_period + length]
    xx = np.dot(x, x)
    ind = max_period - T0
    xy = np.dot(x, pitch_buf[ind:ind + length])
    yy_lookup[0] = xx
    for i in range(yy_lookup.shape[0] - 1):
        ind = max_period - i
        y = pitch_buf[ind:ind + length]
        yy_lookup[i] = np.dot(y, y)
        assert yy_lookup[i] >= 0
    best_xy = xy
    best_yy = yy_lookup[T0]
    g0 = compute_pitch_gain(xx, best_yy, xy)
    best_g = g0

    #TODO(james): document this array
    second_checks = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5]

    #Test harmonics of pitch ind at T0 / k for k between 2 and 13 inclusive
    # stop at 13 since max_period - 13 * min_period < 0
    for k in range(2, 13):
        # round T0 / k to nearest integer
        T = int(round(float(T0) / k))
        if T < min_period:
            break

        # search second period for each T
        # second periods are chosen to give a good coverage across all fractions of T0.
        # ^ this is my guess anyway
        T_2 = int(round(float(T0) * second_checks[k] / k))
        if T_2 > max_period:
            T_2 = T0

        ind = max_period - T
        xy = np.dot(x, pitch_buf[ind: ind + length])
        ind = max_period - T_2
        xy2 = np.dot(x, pitch_buf[ind: ind + length])
        xy_ave = (xy + xy2) / 2.
        yy = yy_lookup[T]
        yy2 = yy_lookup[T_2]
        yy_ave = (yy + yy2) / 2.

        g1 = compute_pitch_gain(xx, yy_ave, xy_ave)

        if abs(T - prev_pitch_period) <= 1:
            prev = prev_pitch_gain
        elif abs(T - prev_pitch_period) <= 2 and 5 * k * k < T0:
            prev = 0.5 * prev_pitch_gain
        else:
            prev = 0

        thresh = max(0.3, 0.7 * g0 - prev)
        if T < 3 * min_period:
            thresh = max(0.4, 0.85 * g0 - prev)

        if T < 2 * min_period:
            thresh = max(0.5, 0.9 * g0 - prev)

        if g1 > thresh:
            best_xy = xy_ave
            best_yy = yy_ave
            best_T = T
            best_g = g1

    best_xy = max(0, best_xy)
    if best_yy <= best_xy:
        pitch_gain = 1
    else:
        pitch_gain = compute_pitch_gain(best_yy, best_yy, best_xy)
    if pitch_gain > best_g:
        pitch_gain = best_g

    if best_T != 0 and best_T != max_period:
        # check around best_T for better xcorr
        xcorr = np.zeros(3)
        #print(best_T)
        for k in range(3):
            ind = max_period - (best_T + k - 1)
            xcorr[k] = np.dot(x, pitch_buf[ind: ind + length])

        # do the same pseudo interpolation as with pitch search
        if xcorr[2] - xcorr[0] > 0.7 * (xcorr[1] - xcorr[0]):
            offset = 1
        elif xcorr[0] - xcorr[2] > 0.7 * (xcorr[1] - xcorr[2]):
            offset = -1
        else:
            offset = 0
    else:
        offset = 0
    pitch_ind = 2 * best_T + offset

    if pitch_ind < PITCH_MIN_PERIOD:
        pitch_ind = PITCH_MIN_PERIOD

    return PITCH_MAX_PERIOD - pitch_ind, pitch_gain

def forward_stft(signal):
    # for some reason he scales the signal down by 960 before putting it into the stft
    scaled_signal = signal / 960.
    stft_settings = VorbisWindowSTFTSettings(sample_rate=48000, window_length=0.02, hop_length=0.01)
    stft_frames = stft(scaled_signal, stft_settings)
    stft_frames_r = np.real(stft_frames[0])
    # not sure why his i is -1 off of ours
    stft_frames_i = np.imag(stft_frames[0]) * -1
    stft_frames = np.stack([stft_frames_r, stft_frames_i], axis=-1)
    return stft_frames

def normalize_corr(pitch_corr, band_energy, pitch_energy):
    epsilon = 0.001
    return pitch_corr / np.sqrt(band_energy * pitch_energy + epsilon)


def compute_pitch_feats_for_frame(
        stft_frame,
        signal_frame,
        band_energies,
        pitch_history,
        prev_index,
        prev_gain
):
    update_pitch_history(pitch_history, signal_frame)
    #pitch_buf is downsampled version of pitch history of length PITCH_BUF_SIZE / 2
    pitch_buf = downsample_pitch_history(pitch_history)
    acorr = compute_autocorr_and_lag_for_lpc(pitch_buf, PITCH_FILTER_ORDER)
    lpc_coeff = lpc(acorr, PITCH_FILTER_ORDER)
    # FIR filter based on lpc coefficients
    filtered_buf = scipy.signal.lfilter(lpc_coeff, [1], pitch_buf)
    pitch_index = pitch_search(filtered_buf)
    pitch_index, pitch_gain = remove_pitch_doubling(
            filtered_buf,
            pitch_index,
            prev_index,
            prev_gain
    )
    pitch_delayed_buf = pitch_history[pitch_index:pitch_index + WINDOW_SIZE]
    pitch_stft = forward_stft(pitch_delayed_buf)
    pitch_stft = pitch_stft[0]
    pitch_energies = compute_band_energy(pitch_stft)
    band_pitch_corr = compute_band_corr(stft_frame, pitch_stft)
    normed_pitch_corr = normalize_corr(band_pitch_corr, band_energies, pitch_energies)
    dct_pitch_corr = dct(normed_pitch_corr)

    # random constant subtraction, not sure what it means
    dct_pitch_corr[0] -= 1.3
    dct_pitch_corr[1] -= 0.9
    # no idea how this supposedly computes the pitch period
    pitch_period_feat = 0.01 * (PITCH_MAX_PERIOD - pitch_index - 300)

    feats = np.hstack([dct_pitch_corr[:NUM_PITCH_FEATS - 1], pitch_period_feat])

    return feats, pitch_index, pitch_gain


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
    flat_signal = (pcm_float_to_i16(signal.flatten()) / 10).astype(np.float32)
    stft_frames = forward_stft(np.hstack([np.zeros(FRAME_SIZE), flat_signal]))
    print("Generating features in python")
    #feats_py, energies = get_features_from_stfts(inputs)
    feats_py, energies = get_features(stft_frames, flat_signal)
    #print('done.')


    #with h5py.File('py_feats.h5', 'w') as hf:
    #    hf.create_dataset('data', data=feats_py)
    #with h5py.File('energies.h5', 'w') as hf:
    #    hf.create_dataset('data', data=energies)
    mse = np.mean(np.square(feats_py - feats))
    print("MSE: {}".format(mse), file=sys.stderr)
    #print(feats_py[:, -1])
    #print(feats[:, -1])
    ##print("MSE energy: {}".format(mse_energy))
