'''
Dear nT analyser, 
if you want to complain please contact: chiara@physik.uzh.ch, gvolta@physik.uzh.ch, kazama@isee.nagoya-u.ac.jp
'''
import datetime
import strax
import straxen
import numba
import numpy as np

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('baseline_window',
                 default=(0,40),
                 help="Window (samples) for baseline calculation."),
    strax.Option('led_window',
                 default=(75, 105),
                 help="Window (samples) where we expect the signal in LED calibration"),
    strax.Option('noise_window',
                 default=(20, 50),
                 help="Window (samples) to analysis the noise"),
    strax.Option('channel_list',
                 default=tuple(range(straxen.n_tpc_pmts)),
                 help="List of PMTs. Defalt value: all the PMTs"),
    )
class LEDCalibration(strax.Plugin):
    """
    Preliminary version, several parameters to set during commisioning.
    LEDCalibration returns: channel, time, dt, lenght, Area, amplitudeLED and amplitudeNOISE.
    The new variables are:
    - Area: Area computed in the given window, averaged over 6 windows that have the same starting sample and different end samples.
    - amplitudeLED: peak amplitude of the LED on run in the given window.
    - amplitudeNOISE: amplitude of the LED on run in a window far from the signal one.
    """

    __version__ = '0.2.4'
    depends_on = ('raw_records',)
    data_kind = 'led_cal' 
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = False

    dtype = [('area', np.float32, 'Area averaged in integration windows'),
             ('amplitude_led', np.float32, 'Amplitude in LED window'),
             ('amplitude_noise', np.float32, 'Amplitude in off LED window'),
             ('channel', np.int16, 'Channel'),
             ('time', np.int64, 'Start time of the interval (ns since unix epoch)'),
             ('dt', np.int16, 'Time resolution in ns'),
             ('length', np.int32, 'Length of the interval in samples')]

    def compute(self, raw_records):
        '''
        The data for LED calibration are build for those PMT which belongs to channel list. 
        This is used for the different ligh levels. As defaul value all the PMTs are considered.
        '''
        mask = np.where(np.in1d(raw_records['channel'], self.config['channel_list']))[0]
        mask &= (raw_records['record_i'] == 0)
        raw_records = raw_records[mask]
        strax.baseline(raw_records, baseline_samples=self.config['baseline_window'][1])

        result = np.zeros(len(raw_records), dtype=self.dtype)

        result['channel'] = raw_records['channel']
        result['time']    = raw_records['time']
        result['dt']      = raw_records['dt']
        result['length']  = raw_records['length']

        led_window = self.config['led_window']
        noise_window = self.config['noise_window']

        # Needed for the SPE computation.
        # Take the maximum in two different regions, where there is
        # the signal and where there is not.
        result['amplitude_led'] = np.max(raw_records['data'][:,
            led_window[0]:led_window[1]], axis=1)
        result['amplitude_noise'] = np.max(raw_records['data'][:,
            noise_window[0]:noise_window[1]], axis=1)

        # Needed for the gain computation.
        # Sum the data in the defined window to get the area.
        # This is done in 6 windows and then averaged.
        left = led_window[0]
        end_pos = led_window[1] + 2*np.arange(6)

        for right in end_pos:
            result['area'] += raw_records['data'][:, left:right].sum(axis=1)
        result['area'] /= len(end_pos)

        return result
