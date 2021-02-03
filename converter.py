'''
Data Converter
cainmagi@gmail.com
Convert the binary file into H5 set (or reverse).
'''

import os
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

import scipy
from scipy import signal
from scipy import interpolate

def ricker(f, length=0.128, dt=0.001):
    t = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

def norm_hist_range(data, bins=300):
    l = np.histogram(data.ravel(), bins=bins, density=True)
    v = np.sort(l[0])[-3]
    x = np.cumsum(l[0])/np.sum(l[0])
    x = np.max([np.abs(l[1][np.argwhere(x > 0.01)[0][0]]), np.abs(l[1][np.argwhere(x > 0.99)[0][0]])])
    return v, x, l

def norm_hist(data):
    v, x, _ = norm_hist_range(data, bins=300)
    plt.hist(data.ravel(), bins=300, density=True)
    plt.xlim([-x, x])
    plt.ylim([0, v])
    
def kl_div(data_1, data_2):
    _, x1, l1 = norm_hist_range(data_1, bins=300)
    _, x2, l2 = norm_hist_range(data_2, bins=300)
    xx = np.max([x1, x2])
    xx = np.linspace(-xx, xx, num=400)
    x1 = np.concatenate([[l1[1][0]-1e-3], l1[1], [l1[1][-1]+1e-3]])
    v1 = np.concatenate([[0.0], l1[0], [0.0, 0.0]])
    #print(x1.shape, v1.shape)
    f = interpolate.interp1d(x1, v1)
    v1 = f(xx)
    v1 = v1/(np.sum(v1)*(xx[1]-xx[0]))
    x2 = np.concatenate([[l2[1][0]-1e-3], l2[1], [l2[1][-1]+1e-3]])
    v2 = np.concatenate([[0.0], l2[0], [0.0, 0.0]])
    f = interpolate.interp1d(x2, v2)
    v2 = f(xx)
    v2 = v2/(np.sum(v2)*(xx[1]-xx[0]))
    kl = -np.sum(v1 * (np.log(v2) - np.log(v1)))
    return kl

class BasicProcessor:
    '''The basic processor class supporting numerical operators.
    '''
    def __init__(self, lam=None):
        if lam is not None:
            self.__lam = lam
        
    def process(self, x):
        if self.__lam is not None:
            return self.__lam(x)
        else:
            return x
    
    @staticmethod
    def __get_process(other):
        proc = getattr(other, 'process')
        if proc and callable(proc):
            return proc
        if callable(other):
            return other
        raise TypeError('The instance {0} is not callable.'.format(other))
    
    def __call__(self, x):
        return self.process(x)
    
    def __sub__(self, other):
        new = type(self)()
        new.process = lambda x: self.process(x) - self.__get_process(other)(x)
        return new
    
    def __add__(self, other):
        new = type(self)()
        new.process = lambda x: self.process(x) + self.__get_process(other)(x)
        return new
    
    def __mul__(self, other):
        new = type(self)()
        new.process = lambda x: self.process(x) * self.__get_process(other)(x)
        return new
    
    def __truediv__(self, other):
        new = type(self)()
        new.process = lambda x: self.process(x) / self.__get_process(other)(x)
        return new
    
    def __floordiv__(self, other):
        new = type(self)()
        new.process = lambda x: np.asarray(self.process(x), dtype=np.int) // np.asarray(self.__get_process(other)(x), dtype=np.int)
        return new
    
    def __pow__(self, other):
        new = type(self)()
        new.process = lambda x: np.pow(self.process(x), self.__get_process(other)(x))
        return new
    
    def __and__(self, other):
        new = type(self)()
        new.process = lambda x: np.logical_and(np.asarray(self.process(x), dtype=np.bool), np.asarray(self.__get_process(other)(x), dtype=np.bool))
        return new
    
    def __or__(self, other):
        new = type(self)()
        new.process = lambda x: np.logical_or(np.asarray(self.process(x), dtype=np.bool), np.asarray(self.__get_process(other)(x), dtype=np.bool))
        return new
    
    def __xor__(self, other):
        new = type(self)()
        new.process = lambda x: np.logical_xor(np.asarray(self.process(x), dtype=np.bool), np.asarray(self.__get_process(other)(x), dtype=np.bool))
        return new
    
class BandPassFilter(BasicProcessor):
    '''Use stationary band-pass filter to process data.
    '''
    
    def __init__(self, **args):
        super(BandPassFilter, self).__init__()
        self.configs = {
            'band_low': 3.0,
            'band_high': 15.0,
            'nyquist': 500,
            'filter': 'butter', # butter, cheby1, cheby2, ellip, bessel
            'out_type': 'ba', # could be 'ba', or 'sos'
            'order': 10,
            'ripple': 1, # Maximum ripple (cheby1, ellip).
            'attenuation': 40, # Minimal attenuation (cheby2, ellip).
            'use_recommend': True
        }
        self.configs.update(args)
        if self.configs['use_recommend']:
            self.use_recommend()
        if self.configs['filter'] == 'butter':
            self.create_butter()
        elif self.configs['filter'] in ('cheby1', 'cheby2'):
            self.create_chebyshev()
        elif self.configs['filter'] == 'ellip':
            self.create_elliptic()
        elif self.configs['filter'] == 'bessel':
            self.create_bessel()
        else:
            raise ValueError('The given filter name is not correct, should be selected from: "butter", "cheby1", "cheby2", "ellip", "bessel".')
            
    def use_recommend(self):
        if self.configs['filter'] in ('butter', 'bessel'):
            self.configs.update({
                'order': 10
            })
        elif self.configs['filter'] == 'cheby1':
            self.configs.update({
                'order': 4,
                'ripple': 5
            })
        elif self.configs['filter'] == 'cheby2':
            self.configs.update({
                'order': 10,
                'attenuation': 40
            })
        elif self.configs['filter'] == 'ellip':
            self.configs.update({
                'order': 4,
                'ripple': 5,
                'attenuation': 40
            })
            
    def create_band(self):
        if self.configs['band_low'] is not None and self.configs['band_high'] is not None:
            wn = (self.configs['band_low'], self.configs['band_high'])
            mode = 'bandpass'
        elif self.configs['band_low'] is not None:
            wn = self.configs['band_low']
            mode = 'highpass'
        elif self.configs['band_high'] is not None:
            wn = self.configs['band_high']
            mode = 'lowpass'
        else:
            raise TypeError('The band-pass filter should not be none.')
        return wn, mode
    
    def create_filter(self, filt):
        if self.configs['out_type'] == 'sos':
            self.filt = functools.partial(signal.sosfilt, sos=filt)
        elif self.configs['out_type'] == 'ba':
            self.filt = functools.partial(signal.lfilter, b=filt[0], a=filt[1])
        else:
            raise ValueError('The out_type is not correct.')
        
    def create_butter(self):
        '''Butterworth filter'''
        wn, mode = self.create_band()
        filt = signal.butter(N=self.configs['order'], Wn=wn, btype=mode, fs=self.configs['nyquist'], output=self.configs['out_type'])
        self.create_filter(filt)
        
    def create_chebyshev(self):
        '''Chebyshev type I/II filter'''
        wn, mode = self.create_band()
        if self.configs['filter'] == 'cheby1':
            filt = signal.cheby1(N=self.configs['order'], rp=self.configs['ripple'], Wn=wn, btype=mode, fs=self.configs['nyquist'], output=self.configs['out_type'])
        elif self.configs['filter'] == 'cheby2':
            filt = signal.cheby2(N=self.configs['order'], rs=self.configs['attenuation'], Wn=wn, btype=mode, fs=self.configs['nyquist'], output=self.configs['out_type'])
        else:
            raise ValueError('Filter type should be "cheby1" or "cheby2".')
        self.create_filter(filt)
        
    def create_elliptic(self):
        '''Elliptic filter'''
        wn, mode = self.create_band()
        filt = signal.ellip(N=self.configs['order'], rp=self.configs['ripple'], rs=self.configs['attenuation'], Wn=wn, btype=mode, fs=self.configs['nyquist'], output=self.configs['out_type'])
        self.create_filter(filt)
        
    def create_bessel(self):
        '''Bessel/Thomson filter'''
        wn, mode = self.create_band()
        filt = signal.bessel(N=self.configs['order'], Wn=wn, btype=mode, norm='phase', fs=self.configs['nyquist'], output=self.configs['out_type'])
        self.create_filter(filt)
        
    def process(self, data):
        data_res = self.filt(x=data)
        return data_res
    
class NonStationaryBandPassFilter(BasicProcessor):
    '''Use non-stationary band-pass filter and taping window to process data.
    '''
    
    def __init__(self, **args):
        super(NonStationaryBandPassFilter, self).__init__()
        self.configs = {
            'length': 2500,
            'band_low': 3.0,
            'band_high': 15.0,
            'nyquist': 500,
            'patch_length': 128,
            'patch_step': 64,
            'kaiser_coef': 1.0,
            'filter': 'none' # butter, cheby1, cheby2, ellip, bessel, none
        }
        self.configs.update(args)
        self.create_patches()
        if self.configs['filter'] not in (None, 'null', 'fft', 'none'):
            fl = BandPassFilter(
                band_low=self.configs['band_low'],
                band_high=self.configs['band_high'],
                nyquist=self.configs['nyquist'],
                filter=self.configs['filter'],
                use_recommend=True
            )
            self.filter_data = fl.process
        else:
            self.filter_data = self.filter_fft
        
    def create_patches(self):
        patch, step = self.configs['patch_length'], self.configs['patch_step']
        length = self.configs['length']
        N = int(np.ceil((length-patch)/step))
        self.patches = list()
        for i in range(N-1):
            self.patches.append((i*step, i*step+patch))
        self.patches.append((length-patch,length))
        self.win = np.kaiser(patch, self.configs['kaiser_coef']*np.pi)
        axis_freq = np.fft.rfftfreq(patch, d=1/self.configs['nyquist'])
        max_freq = axis_freq[-1]
        n_freq = len(axis_freq)
        self.band_low_d = int(np.ceil(self.configs['band_low']/max_freq * n_freq)) if self.configs['band_low'] is not None else None
        self.band_high_d = int(np.ceil(self.configs['band_high']/max_freq * n_freq)) if self.configs['band_high'] is not None else None
        #print(max_freq, patch, self.band_low_d)
        self.win_vec = np.zeros(length)
        for l, r in self.patches:
            self.win_vec[l:r] += self.win
        self.patch_len = patch
        
    def process(self, data):
        data_res = np.zeros_like(data)
        for l, r in self.patches:
            dwin = data[l:r]
            data_res[l:r] += self.filter_data(dwin)
        data_res = data_res / self.win_vec
        return data_res
    
    def filter_fft(self, dwin):
        D = np.fft.rfft(dwin)
        if self.band_low_d is not None:
            D[:self.band_low_d] = 0.0
        if self.band_high_d is not None:
            D[self.band_high_d:] = 0.0
        return np.fft.irfft(D, n=self.patch_len)*self.win
            
class ZeroPadding(BasicProcessor):
    '''Use zero value to pad data, or crop data.
    '''
    def __init__(self, **args):
        super(ZeroPadding, self).__init__()
        self.configs = {
            'padding': (267, -267)
        }
        self.configs.update(args)
        self.pad_begin, self.pad_end = self.create_pad()
        
    def create_pad(self):
        pad = self.configs['padding']
        if isinstance(pad, (list, tuple)):
            pad_begin, pad_end = pad
        elif isinstance(pad, int):
            pad_begin = pad
            pad_end = pad
        else:
            raise TypeError('Zero padding requires two int values (pad_begin, pad_end).')
        if pad_begin is None:
            pad_begin = 0
        if pad_end is None:
            pad_end = 0
        return pad_begin, pad_end
        
    def process(self, x):
        pad_begin, pad_end = self.pad_begin, self.pad_end
        if pad_begin > 0:
            x = np.concatenate([np.zeros(pad_begin), x])
        elif pad_begin < 0:
            x = x[-pad_begin:]
        if pad_end > 0:
            x = np.concatenate([x, np.zeros(pad_end)])
        elif pad_end < 0:
            x = x[:pad_end]
        return x
    
class Lifting(BasicProcessor):
    '''Use log lifting function to enhance the data.
    x = sign(x) * log(1 + a*abs(x))
    '''
    def __init__(self, **args):
        super(Lifting, self).__init__()
        self.configs = {
            'a': 1.0,
            'epsilon': 1e-4,
            'reverse': False,
            'with_norm': False
        }
        self.configs.update(args)
        self.process = self.create_process()
        
    def create_process(self):
        self.a = self.configs['a']
        self.ep = self.configs['epsilon']
        if self.configs['reverse']:
            func = self.process_reverse
        else:
            func = self.process_forward
        if self.configs['with_norm']:
            return functools.partial(self.normalization, func=func, reverse=self.configs['reverse'])
        else:
            return func
        
    def normalization(self, x, func, reverse=False):
        if reverse:
            mx = np.max(np.abs(x))
            return (np.exp(mx)-1.0)*func(x/mx*np.log(1.0+self.a))
        else:
            mx = np.max(np.abs(x))
            return np.log(1.0+mx)*func(x/mx)/np.log(1.0+self.a)
    
    def process_forward(self, x):
        x_abs = np.abs(x)
        x_abs = x_abs
        x_inds = x_abs > self.ep
        res = np.zeros_like(x)
        res[x_inds] = np.sign(x[x_inds]) * np.log(1.0 + self.a * x_abs[x_inds])
        return res
        
    def process_reverse(self, x):
        x_abs = np.abs(x)
        x_inds = x_abs > np.log(1 + self.a * self.ep)
        res = np.zeros_like(x)
        res[x_inds] = np.sign(x[x_inds]) * (np.exp(x_abs[x_inds]) - 1.0)/self.a
        return res
    
class NormalizeMaximal(BasicProcessor):
    '''Normalize with the maximal amplitude.
    '''
    def __init__(self, **args):
        super(NormalizeMaximal, self).__init__()
        self.configs = {
        }
        self.configs.update(args)

    def process(self, x):
        mx = np.max(np.abs(x))
        return x/mx
    
class StackProcessor(BasicProcessor):
    '''Use this class to pack a stack of processors into one.
    '''
    def __init__(self, *processors):
        super(StackProcessor, self).__init__()
        self.processors = processors
        
    def process(self, data):
        for p in self.processors:
            data = p.process(data)
        return data

class CreatorTrue:
    '''Creator for the true dataset.
    Read the true set and the ricker wavelet, and make the training/testing set.
    '''
    
    def __init__(self, root_folder='.'):
        self.root_path = os.path.join(root_folder, 'datasets')
        self.configs = {
            'num_shot': 56,
            'num_recv': 112,
            'num_time': 2500,
            'internal': 10,
            'name_dset': 'data',
            'name_trueset': 'truth',
            'name_wavelet': 'ricker',
            'mode': 'full', # full or lowf
            'remain': False,
            'high_pass': False,
            'low_band': 5.0,
            'high_band': 10.0,
            'nyquist': 500,
            'padding': (267, -267),
            'padding_obj': () # 'dset','truth'
        }
        os.makedirs(self.root_path, exist_ok=True)
        self.obj_low_filter = BandPassFilter(
            length=self.configs['num_time'],
            band_low=None,
            band_high=self.configs['low_band'],
            nyquist=self.configs['nyquist']
        )
        if self.configs['high_pass']:
            self.obj_high_filter = BandPassFilter(
                length=self.configs['num_time'],
                band_low=self.configs['high_band'],
                band_high=None,
                nyquist=self.configs['nyquist']
            )
            self.__to_highfreq = self.obj_high_filter.process
        self.obj_zero_padding = ZeroPadding(
            padding=self.configs['padding']
        )
        self.__to_lowfreq = self.obj_low_filter.process
        self.__pad_trace = self.obj_zero_padding.process
        
    @staticmethod
    def __get_data_test(dset_path, length, offset=0):
        with open(dset_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            print('File num =', size/length/4)
            f.seek(offset*length*4, os.SEEK_SET)
            data = f.read(length*4)
            data = np.frombuffer(data, dtype=np.float32)
            return data
    
    def __plot_data(self, data, freq=False):
        if freq:
            N = len(data)
            data_f = np.fft.rfft(data)
            axis_freq = np.fft.rfftfreq(n=N, d=1/self.configs['nyquist'])
            plt.plot(axis_freq, np.abs(data_f)/N)
        else:
            plt.plot(data)
        
    def test(self, dset_folder, offset=0, freq=False):
        length = self.configs['num_time']
        name_wavelet = os.path.join(dset_folder, self.configs['name_wavelet']+'.BIN')
        if os.path.isfile(name_wavelet):
            wavelet = np.fromfile(name_wavelet, dtype='f4')
        else:
            wavelet = None
        data = self.__get_data_test(os.path.join(dset_folder, self.configs['name_dset']+'.BIN'), length=length, offset=offset)
        truth = self.__get_data_test(os.path.join(dset_folder, self.configs['name_trueset']+'.BIN'), length=length, offset=offset)
        #print(data.shape)
        if 'dset' in self.configs['padding_obj']:
            data = self.__pad_trace(data)
        if 'truth' in self.configs['padding_obj']:
            truth = self.__pad_trace(truth)
        #print(data.shape)
        N = 2
        if self.configs['mode'] == 'lowf' or self.configs['high_pass']:
            N +=1
        if wavelet is not None:
            N +=1
        plt.subplot(N,1,1)
        self.__plot_data(data, freq=freq), plt.xlabel('Input data')
        if freq and self.configs['high_pass'] and self.configs['high_band'] is not None:
            plt.axvline(self.configs['high_band'], color='green', linewidth=1.0)
        plt.subplot(N,1,2)
        self.__plot_data(truth, freq=freq), plt.xlabel('Label data (spikes)')
        s = 3
        if self.configs['high_pass']:
            plt.subplot(N,1,s)
            self.__plot_data(self.__to_highfreq(data), freq=freq), plt.xlabel('High-pass input data (>{0:.2f}Hz)'.format(self.configs['high_band']))
            if freq and self.configs['high_pass'] and self.configs['high_band'] is not None:
                plt.axvline(self.configs['high_band'], color='green', linewidth=1.0)
            s += 1
        else:
            if self.configs['mode'] == 'lowf':
                plt.subplot(N,1,s)
                self.__plot_data(self.__to_lowfreq(truth), freq=freq), plt.xlabel('Low-pass label data (<{0:.2f}Hz)'.format(self.configs['low_band']))
                s += 1
        if wavelet is not None:
            plt.subplot(N,1,s)
            self.__plot_data(wavelet, freq=freq), plt.xlabel('Ricker wavelet')
        plt.gcf().set_size_inches([7.0, 1+N*1.25])
        plt.tight_layout()
        plt.show()
        
    def create_set(self, dset_folder, out_name):
        num_shot, num_recv, num_time = self.configs['num_shot'], self.configs['num_recv'], self.configs['num_time']
        internal = self.configs['internal']
        name_wavelet = os.path.join(dset_folder, self.configs['name_wavelet']+'.BIN')
        out_name = os.path.split(os.path.splitext(out_name)[0])[1]
        if os.path.isfile(name_wavelet):
            wavelet = np.fromfile(name_wavelet, dtype='f4')
        else:
            wavelet = np.nan+np.empty(num_time)
        with open(os.path.join(dset_folder, self.configs['name_dset']+'.BIN'), 'rb') as f,\
                open(os.path.join(dset_folder, self.configs['name_trueset']+'.BIN'), 'rb') as ft,\
                h5py.File(os.path.join(self.root_path, out_name)+'-train.h5', 'w') as ftr, \
                h5py.File(os.path.join(self.root_path, out_name)+'-test.h5', 'w') as fts:
            # Write wavelet
            ftr.create_dataset('wavelet', data=wavelet, dtype=np.float)
            fts.create_dataset('wavelet', data=wavelet, dtype=np.float)
            # Write info
            num_shot_ts = int(np.ceil(num_shot / internal))
            num_shot_tr = num_shot - num_shot_ts
            if not self.configs['remain']:
                num_shot_ts = num_shot
            # Create data
            ftr.create_dataset('X', (num_shot_tr, num_recv, 2*num_time), dtype=np.float, compression='gzip', chunks=(1, 1, 2*num_time))
            fts.create_dataset('X', (num_shot_ts, num_recv, 2*num_time), dtype=np.float, compression='gzip', chunks=(1, 1, 2*num_time))
            ftr['X'].attrs['num_shot'] = num_shot
            fts['X'].attrs['num_shot'] = num_shot
            ftr['X'].attrs['num_shot_tr'] = num_shot_tr
            fts['X'].attrs['num_shot_ts'] = num_shot_ts
            ftr['X'].attrs['num_recv'] = num_recv
            fts['X'].attrs['num_recv'] = num_recv
            ftr['X'].attrs['num_time'] = num_time
            fts['X'].attrs['num_time'] = num_time
            # Create data placeholder
            ftr.create_dataset('len', data=num_recv*np.ones(num_shot_tr), dtype=np.int)
            fts.create_dataset('len', data=num_recv*np.ones(num_shot_ts), dtype=np.int)
            ftr.create_dataset('X_freq', data=np.ones(1), dtype=np.float)
            fts.create_dataset('X_freq', data=np.ones(1), dtype=np.float)
            # Load data
            s_tr = 0
            s_ts = 0
            f.seek(0, os.SEEK_SET)
            ft.seek(0, os.SEEK_SET)
            dset_tr = ftr['X']
            dset_ts = fts['X']
            for n_s in range(num_shot):
                print('Current shot =', n_s)
                for n_r in range(num_recv):
                    data = f.read(num_time*4)
                    data = np.frombuffer(data, dtype=np.float32)
                    data_true = ft.read(num_time*4)
                    data_true = np.frombuffer(data_true, dtype=np.float32)
                    if 'dset' in self.configs['padding_obj']:
                        data = self.__pad_trace(data)
                    if self.configs['high_pass']:
                        data = self.__to_highfreq(data)
                    if 'truth' in self.configs['padding_obj']:
                        data_true = self.__pad_trace(data_true)
                    if self.configs['mode'] == 'lowf':
                        data_true = self.__to_lowfreq(data_true)
                    if n_s % internal != 0:
                        dset_tr[s_tr, n_r, num_time:] = data
                        dset_tr[s_tr, n_r, :num_time] = data_true
                    if (not self.configs['remain']) or (n_s % internal == 0):
                        dset_ts[s_ts, n_r, num_time:] = data
                        dset_ts[s_ts, n_r, :num_time] = data_true
                if n_s % internal != 0:
                    s_tr += 1
                if (not self.configs['remain']) or (n_s % internal == 0):
                    s_ts += 1
                
class RawProcessor:
    '''Pre/post processing for the raw dataset (bin file)
    '''
    
    def __init__(self, proc_func=None, root_folder='.'):
        self.root_path = os.path.join(root_folder, 'datasets')
        self.configs = {
            'num_shot': 56,
            'num_recv': 112,
            'num_time': 2500,
            'nyquist': 500,
        }
        os.makedirs(self.root_path, exist_ok=True)
        self.proc_func = None
        self.load_processor(proc_func)
        
    def load_processor(self, proc_func):
        if proc_func is not None:
            if isinstance(proc_func, (tuple, list)):
                self.proc_func = StackProcessor(*proc_func)
            else:
                self.proc_func = proc_func
        else:
            self.proc_func = lambda x:x
        
    @staticmethod
    def __get_data_test(dset_path, length, offset=0):
        with open(dset_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            print('File num =', size/length/4)
            f.seek(offset*length*4, os.SEEK_SET)
            data = f.read(length*4)
            data = np.frombuffer(data, dtype=np.float32)
            return data
        
    def __plot_data(self, data, freq=False):
        if freq:
            N = len(data)
            data_f = np.fft.rfft(data)
            axis_freq = np.fft.rfftfreq(n=N, d=1/self.configs['nyquist'])
            plt.plot(axis_freq, np.abs(data_f)/N)
        else:
            plt.plot(data)
    
    def test(self, dset_name, offset=0):
        length = self.configs['num_time']
        data = self.__get_data_test(dset_name, length=length, offset=offset)
        proc_data = self.proc_func(data)
        plt.subplot(411)
        self.__plot_data(data, freq=False), plt.xlabel('Data')
        plt.subplot(412)
        self.__plot_data(data, freq=True), plt.xlabel('Data (freq. amp.)')
        plt.subplot(413)
        self.__plot_data(proc_data, freq=False), plt.xlabel('Processed data')
        plt.subplot(414)
        self.__plot_data(proc_data, freq=True), plt.xlabel('Processed data (freq. amp.)')
        plt.gcf().set_size_inches([7.0, 6.0])
        plt.tight_layout()
        plt.show()
        
    def test_shot(self, dset_name, off_shot=0):
        num_shot = self.configs['num_shot']
        num_recv = self.configs['num_recv']
        length = self.configs['num_time']
        with open(dset_name, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            print('File num =', size/length/4)
            f.seek(off_shot*num_recv*length*4, os.SEEK_SET)
            data_shot = []
            data_proc = []
            for i in range(num_recv):
                data = f.read(length*4)
                data = np.frombuffer(data, dtype=np.float32)
                data_shot.append(data)
                data_proc.append(self.proc_func(data))
            data_shot = np.stack(data_shot, axis=1)
            data_proc = np.stack(data_proc, axis=1)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        max_shot = np.percentile(np.abs(data_shot), 99)
        max_proc = np.percentile(np.abs(data_proc), 99)
        plt.subplot(121)
        plt.imshow(data_shot, aspect='auto', cmap='Greys', 
                   norm=mpl.colors.Normalize(vmin=-max_shot, vmax=max_shot, clip=True))
        plt.title('Input, shot {0}'.format(off_shot))
        plt.xlabel('receiver')
        plt.ylabel('time')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(data_proc, aspect='auto', cmap='Greys', 
                   norm=mpl.colors.Normalize(vmin=-max_proc, vmax=max_proc, clip=True))
        plt.title('Processed, shot {0}'.format(off_shot))
        plt.xlabel('receiver')
        fig.set_size_inches([7.5, 5.0])
        plt.tight_layout(rect=[0.0, 0.0, 0.95, 1.0])
        plt.colorbar()
        plt.show()
        
    def processing(self, dset_name):
        out_name = os.path.splitext(dset_name)[0] + '_PROC.BIN'
        num_shot, num_recv, num_time = self.configs['num_shot'], self.configs['num_recv'], self.configs['num_time']
        with open(dset_name, 'rb') as fi,\
                open(out_name, 'wb') as fo:
            for n_s in range(num_shot):
                print('Current shot =', n_s)
                for n_r in range(num_recv):
                    data = fi.read(num_time*4)
                    data = np.frombuffer(data, dtype=np.float32)
                    data_proc = np.asarray(self.proc_func(data), dtype=np.float32, order='C')
                    data_proc = data_proc.tobytes()
                    fo.write(data_proc)
                    
class CompareProcessor:
    '''Compare two datasets (both sets have the same shape).
    '''
    def __init__(self, proc_func1=None, proc_func2=None, root_folder='.'):
        self.root_path = os.path.join(root_folder, 'datasets')
        self.configs = {
            'num_shot': 56,
            'num_recv': 112,
            'num_time': 2500,
            'nyquist': 500,
            'search_step': 20,
            'search_range': (0.2, 20.0)
        }
        os.makedirs(self.root_path, exist_ok=True)
        self.proc_func1 = None
        self.proc_func2 = None
        self.proc_func1 = self.load_processor(proc_func1)
        self.proc_func2 = self.load_processor(proc_func2)
        self.s_step = self.configs['search_step']
        self.s_range = self.configs['search_range']
        
    def load_processor(self, proc_func):
        if proc_func is not None:
            return proc_func
        else:
            return lambda x:x
        
    def get_raw_data(self, dset_name, proc_func):
        num_shot = self.configs['num_shot']
        num_recv = self.configs['num_recv']
        length = self.configs['num_time']
        shot_list = np.linspace(0, num_shot-1, num=8, dtype=np.int)
        with open(dset_name, 'rb') as f:
            data_shot = []
            data_proc = []    
            for off_shot in shot_list:
                f.seek(off_shot*num_recv*length*4, os.SEEK_SET)
                for i in range(num_recv):
                    data = f.read(length*4)
                    data = np.frombuffer(data, dtype=np.float32)
                    data_shot.append(data)
                    data_proc.append(proc_func(data))
            data_shot = np.stack(data_shot, axis=1)
            data_proc = np.stack(data_proc, axis=1)
        return data_shot, data_proc
    
    @staticmethod
    def proc_data(func, data):
        n = data.shape[1]
        res = []
        for i in range(n):
            res.append(func(data[..., i]))
        return np.stack(res, axis=1)
        
    def compare(self, dset_1, dset_2=None):
        if dset_2 is None:
            dset_2 = dset_1
        _, data_proc_1 = self.get_raw_data(dset_1, self.proc_func1)
        _, data_proc_2 = self.get_raw_data(dset_2, self.proc_func2)
        norm_hist(data_proc_1)
        norm_hist(data_proc_2)
        plt.title('KL divergence={0:g}'.format(kl_div(data_proc_1, data_proc_2)))
        plt.show()
        
    def search_lift(self, dset_1, dset_2):
        _, data_proc_1 = self.get_raw_data(dset_1, self.proc_func1)
        data_shot_2, _ = self.get_raw_data(dset_2, self.proc_func2)
        a = self.proc_func1.processors[0].a
        def func(r):
            self.proc_func2.processors[0].a = r*a
            data_proc_2 = self.proc_data(self.proc_func2, data_shot_2)
            kl = kl_div(data_proc_1, data_proc_2)
            return kl
        get = scipy.optimize.brute(func, (self.s_range,), Ns=self.s_step, disp=True, full_output=True, finish=scipy.optimize.fmin)
        print('Solved parameter a={0}'.format(get[0][0]*a))
        self.proc_func2.processors[0].a = get[0][0]*a
                    
if __name__ == '__main__':
    
    from argparse import ArgumentParser
    
    root_dir = r'./datasets_bin'
    
    def double_test(creator, path):
        creator.test(path, offset=0, freq=False)
        creator.test(path, offset=0, freq=True)
        
    def create_true(**kwargs):
        c = CreatorTrue()
        # TDATA_PTL1_DMBP450x160zD10_56S112R_15to3Hz
        # TDATA_DMBP450x160zD10_56S112R_RICK15HZ, TDATA_DMBP450x160zD10_56S112R_15Hz
        dset_name = os.path.splitext(kwargs['setpath'])[0]
        if kwargs['showonly']:
            double_test(c, dset_name)
        else:
            c.create_set(dset_name, dset_name)
    
    def lift_data(**kwargs):
        dset_name = os.path.splitext(kwargs['setpath'])[0]  + '.BIN'
        if kwargs['ltype'] not in ('padinput', 'padtruth', 'input', 'truth'):
            raise ValueError('The input data type should be "input", "padinput", "truth", or "padtruth".')
        procs = []
        if kwargs['ltype'].startswith('pad'):
            zp = ZeroPadding(padding=(267, -267))
            procs.append(zp)
        if kwargs['ltype'].endswith('input'):
            lf = BandPassFilter(
                length=2500,
                band_low=10.0,
                band_high=None,
                nyquist=500.0,
            )
            procs.append(lf)
        ft = Lifting(a=kwargs['a'], reverse=False, with_norm=False)
        procs.append(ft)
        st = StackProcessor(*procs)
        r = RawProcessor(proc_func=st)
        if kwargs['showonly']:
            r.test_shot(dset_name, off_shot=25)
            r.test(dset_name, offset=25*112)
        else:
            r.processing(dset_name)
        
    def lift_data_back(**kwargs):
        dset_name = os.path.splitext(kwargs['setpath'])[0]  + '.BIN'
        ft = Lifting(a=kwargs['a'], reverse=True, with_norm=False) # 0.1, 1.034606
        #nm = NormalizeMaximal()
        r = RawProcessor(proc_func=ft)
        if kwargs['showonly']:
            r.test_shot(dset_name, off_shot=25)
            r.test(dset_name, offset=25*112+33)
        else:
            r.processing(dset_name)
    
    def compare_lift(**kwargs):
        dset_name1 = os.path.splitext(kwargs['setpath'])[0]  + '.BIN'
        dset_name2 = os.path.splitext(kwargs['set2path'])[0] + '.BIN'
        ft1 = Lifting(a=kwargs['a'], reverse=False, with_norm=False)
        ft2 = Lifting(a=kwargs['ainit'], reverse=False, with_norm=False)
        ft_rv = Lifting(a=kwargs['a'], reverse=True, with_norm=False)
        nm = NormalizeMaximal()
        st1 = StackProcessor(ft1, nm)
        st2 = StackProcessor(ft2, nm)
        
        r = RawProcessor(proc_func=st1)
        r.test_shot(dset_name1, off_shot=25)
        r = RawProcessor(proc_func=st2)
        r.test_shot(dset_name2, off_shot=25)
        
        cp = CompareProcessor(proc_func1=st1, proc_func2=st2)
        if not kwargs['showonly']:
            cp.search_lift(dset_name1, dset_name2)
        cp.compare(dset_name1, dset_name2)
        
    parser = ArgumentParser()

    parser.add_argument('-mode',
                        help='Mode of processor. Could be "toh5", "lift", "invlift" or "liftkl"')
    parser.add_argument('-setpath',
                        help='Path for the input dataset.')
    parser.add_argument('-set2path',
                        help='Path for the second input dataset ("liftkl").')
    parser.add_argument('-a', default=0.1, type=float,
                        help='Parameter for the lifting set ("lift", "invlift", "liftkl").')
    parser.add_argument('-ainit', default=1.0, type=float,
                        help='Initialized parameter for the second lifting set ("liftkl").')
    parser.add_argument('-ltype', default='input',
                        help='Could be "input", "padinput", "truth", "padtruth" ("lift", "invlift", "liftkl").')
    parser.add_argument('-showonly', action='store_true',
                        help='Do not actually process, but show one example of the results. This option is used for tuning parameters.')
    args = parser.parse_args()
    args = vars(args)
    
    if args['mode'] is None:
        raise TypeError('Need to specify -mode.')
    
    if args['mode'] == 'toh5':
        create_true(**args)
    elif args['mode'] == 'lift':
        lift_data(**args)
    elif args['mode'] == 'invlift':
        lift_data_back(**args)
    elif args['mode'] == 'liftkl':
        compare_lift(**args)
    else:
        parser.print_help()
