"""Time_eval_receiver_wise.py: File to evaluate the model's prediction results by receiver-wise output a h5 predicted dataset for AGT_FWI_PROJECT2020"""

# Example Usage:  python Time_eval_receiver_wise_alpha.py -model /homelocal/AGT_FWI_2020_Alpha/output/models/CNN19_ResUNet1_Time_Marine_Viking_Train_state_dict.pt -dataset /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/time/Time_Marine_Viking_Test.h5 -output /homelocal/AGT_FWI_2020_Alpha/output/
# python Time_eval_receiver_wise_alpha.py -model ./output2/models/CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ-train_state_dict.pt -dataset ./output/datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ-test.h5 -output ./output2/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# monitor the time for each experiment
import time


def main():
    """ The main function that parses input arguments, read the appropriate
     datasets and model used for evaluation, output a hdf5 which can be used to compare with the ground truth"""
    # Parse input arguments START
    import os
    from argparse import ArgumentParser
    import sys
    import h5py
    from Functions import load_dataset
    from Functions import extract
    from Functions import time_normalize
    import numpy as np
    import torch
    from scipy import fftpack
    from models_zoo_alpha import models
    import struct
    import params as pm

    parser = ArgumentParser()
    parser.add_argument("-model", help="Specify the path of the model for evaluation")
    parser.add_argument("-dataset", help="Specify the path of the dataset for evaluation")
    parser.add_argument("-output", help="Specify the output path for storing the results")
    args = parser.parse_args()

    # Choose model
    if args.model is None:
        sys.exit("Specify the path of the model")
    else:
        path_model = args.model
        name_model = os.path.split(path_model)[1].split('_state_')[0]
        print('name_model: ' + name_model)

    # Choose dataset
    if args.dataset is None:
        sys.exit("Specify the path of the dataset to predict")
    else:
        path_dataset = args.dataset
        name_dataset = os.path.splitext(os.path.split(path_dataset)[1])[0]
        print('name_dataset: ' + name_dataset)

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_output_dataset = os.path.join(path_output, 'datasets')
    os.makedirs(path_output_dataset, exist_ok=True)
    # Parse input arguments END

    # read the dataset to predict
    x, x_freq, length = load_dataset(path_dataset)

    # extract traces
    data_traces = extract(x, length)
    data_traces, normalized_data_buffer = time_normalize(np.array(data_traces))
    data_traces = np.array(data_traces)
    data_test = torch.FloatTensor(data_traces)

    # Load the model for evaluation
    num_input_complex = pm.num_input_complex
    num_output_complex = pm.num_output_complex
    num_input = pm.num_input
    num_output = pm.num_output
    num_receiver = x.shape[1]
    num_shot = x.shape[0]
    x_test_result_Time = np.zeros((num_shot, num_receiver, num_input))
    x_test_result = np.zeros((num_shot, num_receiver, num_input_complex + num_output_complex))
    model, loss_func, optimizer = models(name_model.split('_')[0] + '_' + name_model.split('_')[1])
    model.load_state_dict(torch.load(path_model))
    model.eval()

    # assign GPU
    if pm.gpu_number is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(pm.gpu_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Using Device: {0}, GPU: {1}".format(device, os.environ.get("CUDA_VISIBLE_DEVICES", None)))
    
    data_test = data_test.to(device)
    forward_time = 0
    for i in range(num_shot):
        for j in range(num_receiver):
            num_trace = i * num_receiver + j
            test_input = data_test.data[num_trace][num_output: data_test.shape[1]].unsqueeze(0).unsqueeze(0)
            time_bf_forwar = time.time()
            test_output = model(test_input).cpu().squeeze(0).detach().numpy()
            forward_time = forward_time + (time.time()-time_bf_forwar)
            test_output = test_output * normalized_data_buffer[num_trace]
            x_test_result_Time[i][j][:] = test_output
            test_output_fft = fftpack.fft(test_output)
            sig_scale_fft_extract = test_output_fft[pm.freq_range_1_40[0]:pm.freq_range_1_40[1]]
            complex_sig_fft_trace = []
            for k in range(sig_scale_fft_extract.shape[0]):
                complex_sig_fft_trace = np.append(complex_sig_fft_trace, [sig_scale_fft_extract[k].real])
                complex_sig_fft_trace = np.append(complex_sig_fft_trace, [sig_scale_fft_extract[k].imag])
            x_test_result[i][j][:] = complex_sig_fft_trace

    print("--- %s seconds ---" % forward_time)
    path = os.path.join(path_output_dataset, name_model + '_Predict_' + name_dataset + '_Result.h5')
    # write the train dataset
    length = np.zeros(num_shot).astype(int) + int(num_receiver)
    time_step = pm.time_step  # sample 2 ms
    time_range = time_step*num_input  # signal length 5s at the time domin
    time_vec = np.arange(0, time_range, time_step)
    time_fft = fftpack.fft(time_vec)
    sample_freq = fftpack.fftfreq(time_fft.size, d=time_step)[pm.freq_range_1_40[0]:pm.freq_range_1_40[1]]
    x_freq = sample_freq
    f = h5py.File(path, 'w')
    f.create_dataset('X', data=x_test_result)
    f.create_dataset('X_freq', data=x_freq)
    f.create_dataset('len', data=length)
    f.close()

    # write predicted result as original binary dataset
    path = os.path.join(path_output_dataset, name_model + '_Predict_' + name_dataset + '_Result.bin')
    x_test_result_Time = x_test_result_Time.flatten()
    with open(path, 'wb')as fp:
        for i in range(len(x_test_result_Time)):
            a = struct.pack('f', float(x_test_result_Time[i]))
            fp.write(a)
    fp.close()


if __name__ == "__main__":
    main()
