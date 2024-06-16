# 29/5/2024
# equalizer.py-Compling
# mingliu
# descrption: This script can ...
# input:
# output:

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
# import simpleaudio as sa
# import soundfile as sf


def butterworth_filter(signal, sample_rate, cutoff_frequency, filter_type, order):
    """
    Apply a Butterworth filter (high-pass or low-pass) to the input signal.

    Parameters:
    signal (numpy.ndarray): The input signal to be filtered.
    sample_rate (float): The sampling rate of the input signal (in Hz).
    cutoff_frequency (float): The cutoff frequency of the filter (in Hz).
    filter_type (str): The type of the filter, either 'high' or 'low'.
    order (int): The order of the Butterworth filter.

    Returns:
    numpy.ndarray: The filtered signal.
    """
    nyquist_frequency = sample_rate / 2
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    if filter_type == 'low':
        b, a = scipy.signal.butter(order, normalized_cutoff, btype='low', analog=False, output='ba')
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
    elif filter_type == 'high':
        sos = scipy.signal.butter(order, normalized_cutoff, 'highpass',  output='sos')
        filtered_signal = scipy.signal.sosfilt(sos, signal)
        #b, a = scipy.signal.butter(order, normalized_cutoff, btype='highpass', analog=False, output='ba')
    else:
        raise ValueError("Invalid filter type. Use 'low' or 'high'.")

    #play_sound(filtered_signal,sample_rate)
    return filtered_signal


def read_wav_file(file_path):
    """
    Read a WAV file and return the audio signal and sample rate.

    Parameters:
    file_path (str): The path to the WAV file.

    Returns:
    numpy.ndarray: The audio signal.
    float: The sample rate of the audio signal (in Hz).
    """
    signal, sample_rate = sf.read(file_path)
    return signal, sample_rate



def visualize_spectrum(signal, sample_rate, title):
    """
    Visualize the spectrum of the input signal.

    Parameters:
    signal (numpy.ndarray): The input signal.
    sample_rate (float): The sampling rate of the input signal (in Hz).
    title (str): The title of the plot.
    """
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)

    plt.figure(figsize=(10, 6))
    plt.semilogx(np.abs(frequencies), 20 * np.log10(np.abs(fft_signal)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(title)
    plt.grid(True)
    plt.show()


def play_sound(signal, sample_rate):
    """
    Play the input signal through the default audio device.

    Parameters:
    signal (numpy.ndarray): The input signal to be played.
    sample_rate (float): The sampling rate of the input signal (in Hz).
    """
    audio = (signal * (2 ** 15 - 1)).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()


def export_to_wav(signal, sample_rate, filename):
    """
    Export the input signal to a WAV file.

    Parameters:
    signal (numpy.ndarray): The input signal to be exported.
    sample_rate (float): The sampling rate of the input signal (in Hz).
    filename (str): The filename for the exported WAV file.
    """
    audio = (signal * (2 ** 15 - 1)).astype(np.int16)
    sf.write(filename, audio, sample_rate)
    print(f"Saved to {filename}")


def boost_low_frequencies(audio, fs, cutoff_freq, boost_db):
    """
    Boost the energy of the low-frequency components in the input audio signal using a high-pass filter.

    Args:
        audio (numpy.ndarray): The input audio signal.
        fs (int): The sampling rate of the audio signal (in Hz).
        cutoff_freq (float): The cutoff frequency of the high-pass filter (in Hz).
        boost_db (float): The amount of boost (in decibels) to apply to the low-frequency components.

    Returns:
        numpy.ndarray: The audio signal with the low-frequency components boosted.
    """
    # Design a high-pass Butterworth filter
    nyquist = fs / 2
    order = 4  # Order of the Butterworth filter, doesn't matter here
    normalized_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply the low-pass filter to the audio signal
    filtered_audio = scipy.signal.filtfilt(b, a, audio) # this will remove all high frequency components
    high_fre = audio - filtered_audio # high-frequency components
    #export_to_wav(high_fre,sample_rate=sample_rate,filename='high_fre_com.wav')
    #play_sound(filtered_audio,sample_rate=sample_rate)

    # Calculate the boost factor
    boost_factor = 10 ** (boost_db / 20)
    low_fre = boost_factor * filtered_audio # emphasized low frequency components
    #export_to_wav(low,sample_rate=sample_rate,filename='low_boosted.wav')
    # recombine high and low frequency components.
    boosted_audio = low_fre + high_fre

    return boosted_audio


if __name__ == "__main__":
    # Example usage
    sound, sample_rate = read_wav_file('sa.wav')
    #play_sound(sound,sample_rate)


    # if pre-emphasis
    sound = boost_low_frequencies(sound,fs=sample_rate,cutoff_freq=2000,boost_db=12)
    #play_sound(sound,sample_rate)
    #export_to_wav(sound,sample_rate,"preemphasis.wav")

    # Low-pass filter
    low_pass_cutoff = 4000 # Hz
    low_pass_order = 2 # 12dB/octave
    low_pass_signal = butterworth_filter(sound, sample_rate, low_pass_cutoff, 'low', low_pass_order)

    # High-pass filter
    high_pass_cutoff =4000  # Hz
    high_pass_order = 2 # 12dB/octave
    high_pass_signal = butterworth_filter(sound, sample_rate, high_pass_cutoff, 'high', high_pass_order)

    # Visualize the spectrum of the input signal (white noise)
    #visualize_spectrum(sound, sample_rate, 'Spectrum of Original sound')

    # Visualize the spectrum of the low-pass filtered signal
    #visualize_spectrum(low_pass_signal, sample_rate, 'Spectrum of Low-Pass Filtered Signal')

    # Visualize the spectrum of the high-pass filtered signal
    #visualize_spectrum(high_pass_signal, sample_rate, 'Spectrum of High-Pass Filtered Signal')

    # Play the original white noise
    #print("Playing original white noise...")
    #play_sound(sound, sample_rate)

    # Export the signals to WAV files
    #export_to_wav(sound, sample_rate, 'sound.wav')
    export_to_wav(low_pass_signal, sample_rate, 'low_pass_signal.wav')
    export_to_wav(high_pass_signal, sample_rate, 'high_pass_signal.wav')

    ###

    '''1st order Butterworth filter: 6 dB/octave
    2nd order Butterworth filter: 12 dB/octave
    3rd order Butterworth filter: 18 dB/octave
    4th order Butterworth filter: 24 dB/octave
    5th order Butterworth filter: 30 dB/octave
    6th order Butterworth filter: 36 dB/octave
    7th order Butterworth filter: 42 dB/octave
    8th order Butterworth filter: 48 dB/octave'''