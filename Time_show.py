'''
Use these codes to view datasets.
'''

import os, sys
from argparse import ArgumentParser

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.restoration import unwrap_phase

def make_segmented_cmap(wrap=0): 
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    if isinstance(wrap, int):
        if wrap == 0:
            wrap = None
        else:
            wrap = (wrap, wrap)
    if wrap is not None:
        mapind = [white, red, black, blue] * (wrap[0] // 2) + [white]
        if wrap[0] % 2 != 0:
            mapind = [black, blue] + mapind
        mapind = mapind + [red, black, blue, white] * (wrap[1] // 2)
        if wrap[1] % 2 != 0:
            mapind = mapind + [red, black]
        N = 256*(wrap[1] + wrap[0])
    else:
        mapind = [black, blue, white, red, black]
        N = 256
    anglemap = mpl.colors.LinearSegmentedColormap.from_list(
        'anglemap', mapind, N=N, gamma=1)
    return anglemap

def draw_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.08)
    fig.colorbar(im, cax=cax, orientation='vertical')

class Loader:
    
    def __init__(self):
        self.configs = {
            'num_shot': 56,
            'num_recv': 112,
            'num_time': 2500,
            'nyquist': 500
        }
        
    def get_one_shot(self, dset_name, n_shot):
        dset_name = os.path.splitext(dset_name)[0] + '.BIN'
        num_recv, num_time = self.configs['num_recv'], self.configs['num_time']
        with open(dset_name, 'rb') as f:
            f.seek(n_shot*num_recv*num_time*4, os.SEEK_SET)
            data_shot = f.read(num_recv*num_time*4)
            data_shot = np.frombuffer(data_shot, dtype=np.float32)
            data_shot = np.reshape(data_shot, (num_recv, num_time))
            data_shot = np.transpose(data_shot)
        return data_shot
    
    def get_one_trace(self, dset_name, n_shot, n_recv):
        dset_name = os.path.splitext(dset_name)[0] + '.BIN'
        num_recv, num_time = self.configs['num_recv'], self.configs['num_time']
        with open(dset_name, 'rb') as f:
            f.seek((n_shot*num_recv + n_recv)*num_time*4, os.SEEK_SET)
            data = f.read(num_time*4)
            data = np.frombuffer(data, dtype=np.float32)
        return data
    
    def get_phase(self, dset_name, freq=3.0):
        num_shot, num_time = self.configs['num_shot'], self.configs['num_time']
        n_f = np.nonzero(np.fft.rfftfreq(n=num_time, d=1/self.configs['nyquist']) > freq)[0][0]
        data_phase = []
        for s in range(num_shot):
            data = self.get_one_shot(dset_name, s)
            data = np.fft.rfft(data, axis=0)
            data = np.angle(data[n_f, ...])
            data_phase.append(data)
        data_phase = np.stack(data_phase, axis=0)
        return data_phase
    
class Drawer(Loader):
    
    def draw_shot_view(self, dpath, dpath_true=None, shot_num=25):
        if dpath_true is not None:
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            
            data = self.get_one_shot(dpath, shot_num)
            max_v = np.percentile(np.abs(data), 99)
            norm = mpl.colors.Normalize(vmin=-max_v, vmax=max_v, clip=True)
            im = axes[0].imshow(data, aspect='auto', norm=norm, cmap='Greys')
            axes[0].set_title('Prediction'), axes[0].set_xlabel('receiver'), axes[0].set_ylabel('time')
            draw_colorbar(fig, axes[0], im)
            
            data_true = self.get_one_shot(dpath_true, shot_num)
            max_v = np.percentile(np.abs(data_true), 99)
            norm = mpl.colors.Normalize(vmin=-max_v, vmax=max_v, clip=True)
            im = axes[1].imshow(data_true, aspect='auto', norm=norm, cmap='Greys')
            axes[1].set_title('Ground truth'), axes[1].set_xlabel('receiver')
            draw_colorbar(fig, axes[1], im)
            
            data_res = data - data_true
            max_v = max(1.0, np.percentile(np.abs(data_res), 99))
            norm = mpl.colors.Normalize(vmin=-max_v, vmax=max_v, clip=True)
            im = axes[2].imshow(data_res, aspect='auto', norm=norm, cmap='Greys')
            axes[2].set_title('Residual'), axes[2].set_xlabel('receiver')
            draw_colorbar(fig, axes[2], im)
        else:
            n = 3
            fig, axes = plt.subplots(1, n, sharex=True, sharey=True)
            n_list = np.sort(np.random.RandomState(1000).randint(0, self.configs['num_shot'], size=n))
            for i, n_shot in enumerate(n_list):
                data = self.get_one_shot(dpath, n_shot)
                max_v = np.percentile(np.abs(data), 99)
                norm = mpl.colors.Normalize(vmin=-max_v, vmax=max_v, clip=True)
                im = axes[i].imshow(data, aspect='auto', norm=norm, cmap='Greys')
                axes[i].set_title('Shot {0}'.format(n_shot)), axes[i].set_xlabel('receiver')
                draw_colorbar(fig, axes[i], im)
                if i == 0:
                    axes[i].set_ylabel('time')
        plt.gcf().set_size_inches([10.0, 5.0])
        plt.tight_layout()
        plt.show()
        
    def draw_traces_view(self, dpath, dpath_true=None, t_range=(0, None)):
        n = 5
        with_truth = dpath_true is not None
        t_range = (0 if t_range[0] is None else t_range[0], self.configs['num_time'] if t_range[1] is None else t_range[1])
        fig, axes = plt.subplots(1, n, sharex=True, sharey=True)
        st = np.random.RandomState(1000)
        n_shot_list = np.sort(st.randint(0, self.configs['num_shot'], size=n))
        n_recv_list = st.randint(0, self.configs['num_recv'], size=n)
        time_axis = np.arange(*t_range)
        for i, (n_shot, n_recv) in enumerate(zip(n_shot_list, n_recv_list)):
            data = self.get_one_trace(dpath, n_shot, n_recv)
            axes[i].plot(data[t_range[0]:t_range[1]], time_axis, label='Prediction')
            if with_truth:
                data = self.get_one_trace(dpath_true, n_shot, n_recv)
                axes[i].plot(data[t_range[0]:t_range[1]], time_axis, label='Ground truth')
            axes[i].set_title('Trace ({0}, {1})'.format(n_shot, n_recv))
            axes[i].invert_yaxis()
            if i == 0:
                axes[i].set_ylabel('time')
                if with_truth:
                    plt.figlegend(*axes[i].get_legend_handles_labels(), ncol=2, loc='lower center')
        plt.gcf().set_size_inches([9.0, 5.0])
        plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
        plt.show()
        
    @staticmethod
    def _draw_im_wcolor(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='8%', pad=0.08)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
    @staticmethod
    def _draw_phase_im_unwrap(data_phase, fig, ax):
        data_phase = unwrap_phase(data_phase, wrap_around=False)
        wrap = (int(np.ceil(-np.amin(data_phase)/np.pi)), int(np.ceil(np.amax(data_phase)/np.pi)))
        ax.imshow(data_phase, cmap=make_segmented_cmap(wrap=wrap), vmin=-wrap[0]*np.pi, vmax=wrap[1]*np.pi,
            interpolation='bilinear', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='8%', pad=0.08)
        fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi, clip=False), cmap=make_segmented_cmap(wrap=0)), cax=cax, ax=ax)
        
    @staticmethod
    def _draw_phase_im_plain(data_phase, fig, ax, residual=False):
        max_v = max(1.0, np.percentile(np.abs(data_phase), 99))
        norm = mpl.colors.Normalize(vmin=(-max_v if not residual else 0.0), vmax=max_v, clip=True)
        im = ax.imshow(data_phase, aspect='auto', norm=norm, cmap='viridis', interpolation='bilinear')
        draw_colorbar(fig, ax, im)
        
    @classmethod
    def _draw_phase_im(cls, data_phase, fig, ax):
        cls._draw_phase_im_plain(data_phase, fig, ax)
        
    def draw_phase_view(self, dpath, dpath_true=None, freq=3.0):
        if dpath_true is not None:
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            
            data = self.get_phase(dpath, freq=freq)
            self._draw_phase_im(data, fig, axes[0])
            axes[0].set_title('Prediction'), axes[0].set_xlabel('receiver'), axes[0].set_ylabel('shot')
            #self._draw_im_wcolor(fig, axes[0], im)
            
            data_true = self.get_phase(dpath_true, freq=freq)
            self._draw_phase_im(data_true, fig, axes[1])
            axes[1].set_title('Ground truth'), axes[1].set_xlabel('receiver')
            
            data_res = np.abs(data - data_true)
            data_res = np.minimum(np.minimum(data_res, np.abs(data_res - 2*np.pi)), np.abs(data_res + 2*np.pi))
            self._draw_phase_im_plain(data_res, fig, axes[2], residual=True)
            axes[2].set_title('Residual'), axes[2].set_xlabel('receiver')
            plt.gcf().set_size_inches([9.0, 5.0])
        else:
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
            data = self.get_phase(dpath, freq=freq)
            self._draw_phase_im(data, fig, ax)
            ax.set_xlabel('receiver'), ax.set_ylabel('shot')
            plt.gcf().set_size_inches([4.0, 5.0])
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-mode',
                        help='Mode for showing datasets. Could be "shot", "phase", or "trace"')
    parser.add_argument('-setpath',
                        help='Paths for the dataset.')
    parser.add_argument('-trueset',
                        help='Path for the true set. If given, would make comparison.')
    parser.add_argument('-shotnum', type=int, default=25,
                        help='Shot number when setting the -trueset for -mode shot.')
    args = parser.parse_args()
    
    ld = Drawer() # Set up the loader.
    
    if args.mode == 'shot':
        ld.draw_shot_view(args.setpath, args.trueset, shot_num=args.shotnum)
    elif args.mode == 'phase':
        ld.draw_phase_view(args.setpath, args.trueset, freq=3.0)
    elif args.mode == 'trace':
        ld.draw_traces_view(args.setpath, args.trueset, t_range=(0,None))
    