"""dataset_generator_alpha.py: File to convert the Binary dataset to hdf5 dataset for AGT_FWI_PROJECT2020 training"""

# Example Usage: python dataset_generator_alpha.py -path_wavelet /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/datasets/DM_BPmigration94/Ricker15Hz5s.dt2ms.BIN -scale_mode no -split True -path_data /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/datasets/DM_BPmigration94/TDATA_BPmigration94samp1_400x160zD10_200S200R_RICK15HZ.SCT.dt2ms.BIN -output /homelocal/AGT_FWI_2020/output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# monitor the time for this script running
import time

start_time = time.time()


def main():
    """ The main function that parses input arguments, read the appropriate
     dataset and wavelet binary file as inputs, preprocess the dataset and write to hdf5 file for training"""
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    import struct
    from scipy import fftpack
    import h5py

    parser = ArgumentParser()

    parser.add_argument("-path_wavelet", help="Specify the path of the wavelet BIN file")
    parser.add_argument("-scale_mode", help="Specify the way to scale the time data")
    parser.add_argument("-split", help="Specify if you want to split the dataset to train or test")
    parser.add_argument("-path_data", help="Specify the path of the data BIN file")
    parser.add_argument("-output", help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Choose wavelet binary file
    if args.path_wavelet is None:
        sys.exit("Specify the path of the wavelet BIN file")
    else:
        path_wavelet = args.path_wavelet
        name_wavelet = path_wavelet.split('/')[-1].split('.')[0]
        print('wavelet: ' + name_wavelet)

    # Choose scale mode: linear, time power ...
    if args.scale_mode is None:
        sys.exit("Specify the way to scale the time data")
    else:
        scale_mode = args.scale_mode
        print('scale_mode: ' + scale_mode)

    # Choose scale mode: linear, time power ...
    if args.split is None:
        sys.exit("Specify if you want to split the dataset to train and test")
    else:
        split = args.split
        if split == 'False':
            split = False
        elif split == 'True':
            split = True
        print('split: ' + str(split))

    # Choose binary dataset file
    if args.path_data is None:
        sys.exit("Specify the path of the BIN dataset file")
    else:
        path_data = args.path_data
        name_data = path_data.split('/')[-1].split('.')[0]
        print('name_data: ' + name_data)

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_output_dataset = path_output + 'datasets/'

    # read the ricker wavelet
    # file_ricker = open(path_wavelet, "rb")
    # Ricker15Hz5s = []
    time_step = 0.004  # sample interval 2 ms
    time_range = 6  # signal length is 5s
    time_vec = np.arange(0, time_range, time_step)
    # for i in range(int(time_range / time_step)):
    #     Ricker15Hz5s.append(struct.unpack('f', file_ricker.read(4)))
    # Ricker15Hz5s = np.array(Ricker15Hz5s)
    # ricker = Ricker15Hz5s.flatten()
    # file_ricker.close()

    # FFT(wavelet)
    # ricker_fft = fftpack.fft(ricker)
    # sample_freq = fftpack.fftfreq(ricker_fft.size, d=time_step)[5:201]  # 1-40Hz frequency components

    # patch for the Viking
    ricker_fft = fftpack.fft(time_vec)
    sample_freq = fftpack.fftfreq(ricker_fft.size, d=time_step)[6:301] # [6:301] 1-50Hz

    num_shot = 248
    num_receiver = 120

    # # read the total binary dataset to an 1 d numpy array
    # num_shot = int(name_data.split('_')[-2].split('S')[0])
    # num_receiver = int(name_data.split('_')[-2].split('S')[1].split('R')[0])
    len_signal = time_vec.shape[0]  # time_vec.shape[0] is the length of one trace/ seismic signal
    file_data = open(path_data, "rb")

    DATA = []
    for i in range(num_shot * num_receiver * len_signal):
        DATA.append(struct.unpack('f', file_data.read(4)))
    DATA = np.array(DATA)
    data = DATA.flatten()

    # Calculate the X (matrix stores all dataset information)
    X = []
    # Convert dataset to format of X.shape = (num_shot, num_receiver, len_trace(1Hz_Real, 1Hz_Img, ... , 40Hz_Real, 40Hz
    # _Img))
    for i in range(num_shot):
        X.append([])
        for j in range(num_receiver):
            X[i].append([])
            # extract trace
            sig_trace = data[
                        i * num_receiver * len_signal + j * len_signal: i * num_receiver * len_signal + j * len_signal + len_signal]

            # choose Scale mode
            if scale_mode == 'linear':
                scale_constant = time_vec
            # default no scale
            elif scale_mode == 'no':
                scale_constant = 1
            # T power scale
            else:
                scale_constant = time_vec ** int(scale_mode.split('_')[1])
            # shift=>scale ! cancel shift
            sig_fft = fftpack.fft(sig_trace)
            sig_shift = fftpack.ifft(sig_fft / ricker_fft)
            sig_scale = sig_trace * scale_constant
            sig_scale_fft = fftpack.fft(sig_scale)
            # extract 1-40 hz data for each trace
            sig_scale_fft_extract = sig_scale_fft[6:301] # remeind to chang back to [5:201] which represent 1-40 for Marmousi etc.
            complex_sig_fft_trace = []
            for k in range(sig_scale_fft_extract.shape[0]):
                complex_sig_fft_trace = np.append(complex_sig_fft_trace, [sig_scale_fft_extract[k].real])
                complex_sig_fft_trace = np.append(complex_sig_fft_trace, [sig_scale_fft_extract[k].imag])
            X[i][j].append(complex_sig_fft_trace)

    X = np.array(X).squeeze(2)
    # Write the dataset to HDF5 file
    name_data = 'Time_Marine_Viking'

    # f = h5py.File(path_output_dataset + name_data + '.h5', 'w')
    # x_freq = sample_freq
    # length = np.zeros(num_shot).astype(int) + int(num_receiver)
    # f.create_dataset('X', data=X)
    # f.create_dataset('X_freq', data=x_freq)
    # f.create_dataset('len', data=length)
    # f.close()
    if split == True:
        X_train = []
        X_test = []
        for i in range(num_shot):
            if i % 8 == 0:
                X_train.append(X[i])
                print('write ' + str(i) + ' shot to train')
            else:
                X_test.append(X[i])

        path_train = path_output_dataset + name_data + '_Train_8.h5'
        # write the train dataset
        f = h5py.File(path_train, 'w')
        x_freq = sample_freq
        length = np.zeros(len(X_train)).astype(int) + int(num_receiver)
        f.create_dataset('X', data=X_train)
        f.create_dataset('X_freq', data=x_freq)
        f.create_dataset('len', data=length)
        f.close()

        path_test = path_output_dataset + name_data + '_Test.h5'
        # write the test dataset
        f = h5py.File(path_test, 'w')
        x_freq = x_freq
        length = np.zeros(len(X_test)).astype(int) + int(num_receiver)
        f.create_dataset('X', data=X_test)
        f.create_dataset('X_freq', data=x_freq)
        f.create_dataset('len', data=length)
        f.close()
    else:
        pass


if __name__ == "__main__":
    main()
