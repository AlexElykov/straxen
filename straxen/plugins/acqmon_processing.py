import strax
import straxen
import numba
import numpy as np
from numba import njit
from immutabledict import immutabledict

export, __all__ = strax.exporter()

#### Veto hardware ####:
# V1495 busy veto module: 
# Generates a 25 ns NIM pulse whenever a veto begins and a 25 ns NIM signal when it ends. 
# A new start signal can occur only after the previous busy instance ended.
# 1ms (1e6 ns) - minimum busy veto length, or until the board clears its memory
    
# DDC10 High Energy Veto:
# 10ms (1e7 ns) - fixed HE veto length in XENON1T DDC10, 
# in XENONnT it will be calibrated based length of large S2 SE tails
# The start/stop signals for the HEV are generated by the V1495 board
  

@strax.takes_config(
    strax.Option('max_veto_gap', default = int(5e8),
                 help = 'Maximum separation between veto stop and start pulses [ns]'),
    strax.Option('channel_map', track = False, type = immutabledict, 
                 help = 'frozendict mapping subdetector to (min, max)'
                      'channel number.'),
    strax.Option('run_start_time', type = float, track = False, 
                 default = int(0), help='time of start run (s since unix epoch)')
)

@export
class VetoIntervals(strax.OverlapWindowPlugin):
    """ Find pairs of veto start and veto stop signals and the veto duration between them
    busy_*  <= V1495 busy veto for tpc channels
    bb_*    <= V1495 busy veto for high energy tpc channels
    hev_*   <= DDC10 hardware high energy veto
    """
        
    __version__ ='0.0.19'
    depends_on = ('raw_records_aqmon') 
    provides  = ('veto_intervals')
    data_kind = ('veto_intervals')


    aqmon_channel_names = ('sum_wf','m_veto_sync',
                           'hev_stop', 'hev_start', 
                           'bb_stop','bb_start',
                           'busy_stop', 'busy_start')
    
    tmp_dtype = strax.time_fields + [(('Interval between start and stop in [ns]', 'interval'), np.int64)]

    def infer_dtype(self):
        dtype = [(('faux time [ns]', 'time'), np.int64)]
        dtype += [(('faux endtime [ns]', 'endtime'), np.int64)]
        for veto in ('busy_','hev_','bb_'):
            dtype += [(('Start '+ veto +'veto time since unix epoch [ns]', veto + 'time'), np.int64)]
            dtype += [(('Stop '+ veto +'veto time since unix epoch [ns]', veto + 'endtime'), np.int64)]
            dtype += [(('Duration of '+ veto +'veto time since unix epoch [ns]', veto + 'interval'), np.int64)]
        return dtype
    
    def setup(self):
        self.channel_range = self.config['channel_map']['aqmon']
        self.channel_numbers = np.arange(self.channel_range[1]-7,self.channel_range[1] + 1, 1)
        self.channel_map = dict(zip(self.aqmon_channel_names,self.channel_numbers))
        self.t0 = self.config['run_start_time']
        self.veto_names = ['busy_','bb_','hev_']
        return self.channel_map
    
    def get_window_size(self):
        # Give a very wide window
        return (self.config['max_veto_gap']*100)
    
    def compute(self, raw_records_aqmon): 
        rec = strax.raw_to_records(raw_records_aqmon)
        strax.zero_out_of_bounds(rec)
        strax.baseline(rec, baseline_samples=10, flip=True)
        hits = strax.find_hits(rec)
        
        vetos = dict()
        res = dict()
        for i, veto in enumerate(self.veto_names):
            channels = channel_select(hits, self.channel_map[veto + 'stop'], self.channel_map[veto + 'start'])
            vetos[veto + 'veto'] = self.merge_vetos(channels, gap = self.config['max_veto_gap'],\
                                           dtype = self.tmp_dtype, t = 0)
        
        
        result = strax.dict_to_rec(vetos)

        # Populate fake 'time' and 'endtime' by using the time and endime of the veto singals
        for name in self.veto_names:
            if len(result[name + 'veto']['time']) > 0:
                res = dict(
                    time = rec['time'][:len(result[name + 'veto']['time'])],
                    endtime = strax.endtime(rec)[:len(result[name + 'veto']['time'])]) 
        
        for v in self.veto_names:
            for t in ['time', 'endtime', 'interval']:
                res[v + t] = result[v + 'veto'][t]       
        
        return res
    
    @staticmethod
    def merge_vetos(channels, gap, dtype, t):
        if len(channels):
            start, stop = strax.find_peak_groups(channels, gap_threshold = gap)
            result_temp = np.zeros(len(start),dtype=dtype)
            result_temp['time'] = start 
            result_temp['endtime'] = stop
            result_temp['interval'] = stop - start
            # Small hack to remove times that were not properly merged inside the window
            # This also will remove any busies that have their t_veto_stop > t_run_end
            result = result_temp[result_temp['interval'][:] > 10]
        else:
            result = np.zeros(1,dtype=dtype)
            result['time'] =  t
            result['endtime'] = t
            result['interval'] = t
        return result
            
@numba.njit
def _mask(x, mask):
    return x[mask]

@export
@numba.njit
def channel_select(rr, ch_stop, ch_start):
    """Return """
    return _mask(rr, (rr['channel'] >= ch_stop) & (rr['channel'] <= ch_start)) 