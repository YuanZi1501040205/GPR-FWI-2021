"""Time_dataset_generator_alpha.py: File to preprocess the Binary dataset to h5 dataset for training AGT_FWI_PROJECT2020 Time Domain"""

# python Time_dataset_generator_alpha.py -split True -path_data /home/yzi/research/GPR-FWI-2021/geo_model/hdf5 -output ./output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# monitor the time for each experiment
import time

start_time = time.time()


def main():
    """ The main function that parses input arguments, read the appropriate
     dataset and wavelet as inputs, and preprocess the dataset write to the h5 file for training, serve for time domain
     """
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    import struct
    from scipy import fftpack
    import h5py
    import os
    from Functions import butter_highpass_filter

    parser = ArgumentParser()

    parser.add_argument("-split", help="Specify if you want to split the dataset to train and test")
    parser.add_argument("-path_data", help="Specify the path of the data BIN file")
    parser.add_argument("-path_label", help="Specify the path of the label data BIN file")
    parser.add_argument("-output", help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Choose split mode
    if args.split is None:
        sys.exit("Specify if you want to split the dataset to train or test")
    else:
        split = args.split
        if split == 'False':
            split = False
        elif split == 'True':
            split = True
        print('split: ' + str(split))

    # Choose dataset bin file
    if args.path_data is None:
        sys.exit("Specify the path of the data BIN file")
    else:
        path_data = args.path_data
        name_data = 'cross_well_geo_model_1'
        print('name_data: ' + name_data)

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_output_dataset = path_output + 'datasets/'

    time_step = 2.357934449422306e-11
    time_range = 10e-8  # signal length 5s at the time domin
    time_vec = np.arange(0, time_range, time_step)
    fs = 1 / time_step
    sample_freq = fftpack.fftfreq(fftpack.fft(time_vec).size, d=time_step)


    # patch for gpr cross well
    num_shot = 13
    num_receiver = 13

    len_signal = time_vec.shape[0]  # time_vec.shape[0] is the length of one trace/ seismic signal

    # read the all shots from the folder
    DATA = np.zeros((num_shot, num_receiver, 2*len_signal))
    for _ in os.listdir(path_data):
        shot_index = int(_.split('.')[0].split('_shot_')[-1])
        f = h5py.File(path_data + '/' + _, 'r')
        nrx = f.attrs['nrx']
        shot_gather = []
        for __ in range(nrx):
            trace = np.array(f['rxs']['rx' + str(__ + 1)]['Ez'])
            lowcut = 50e6 # 50 Mhz
            trace_input = butter_highpass_filter(trace, lowcut, fs, order=5)
            shot_gather.append(np.concatenate((trace, trace_input)))
        DATA[shot_index] = np.array(shot_gather)



    # Write the dataset to HDF5 file
    f = h5py.File(path_output_dataset + 'Time_' + name_data + '.h5', 'w')
    x_freq = time_vec  # in Time case, replace the frequency component to signal's time length
    length = np.zeros(num_shot).astype(int) + int(num_receiver)
    f.create_dataset('X', data=DATA)
    f.create_dataset('X_freq', data=x_freq)
    f.create_dataset('len', data=length)
    f.close()
    if split:
        X_train = []
        X_test = []
        for i in range(num_shot):
            if i % 2 == 0:
                X_train.append(DATA[i])
                print('write ' + str(i) + ' shot to train')
            else:
                X_test.append(DATA[i])

        path_train = path_output_dataset + 'Time_' + name_data + '_Train.h5'
        # write the train dataset
        f = h5py.File(path_train, 'w')
        length = np.zeros(len(X_train)).astype(int) + int(num_receiver)
        f.create_dataset('X', data=X_train)
        f.create_dataset('X_freq', data=x_freq)
        f.create_dataset('len', data=length)
        f.close()

        path_test = path_output_dataset + 'Time_' + name_data + '_Test.h5'
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
