"""Time_eval_alpha.py: File to perform the evaluation for models trained by the Time_train_alpha.py file TIME"""

# Example Usage: python Time_eval_nolabel_alpha.py -test /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/time/Time_Marine_Viking_Test.h5 -model CNN19_ResUNet1 -model_path /homelocal/AGT_FWI_2020_Alpha/output/models/CNN19_ResUNet1_Time_Marine_Viking_Train_state_dict.pt -output /homelocal/AGT_FWI_2020_Alpha/output/
#python Time_eval_nolabel_alpha.py -test ./output/datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-test.h5 -model CNN19_ResUNet1 -model_path ./output2/models/CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-train_state_dict.pt -output ./output2

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "2.0.0"


def main():
    """ The main function that parses input arguments, load the appropriate
     Neural Networks models and chose test dataset' paths and the output path.
     Output n Predicted Traces figures and receiver-wise, shot-wise, frequency-wise evaluation
     statistic result figures to the output folder"""

    # Parse input arguments
    import os
    from argparse import ArgumentParser
    import sys
    from Functions import load_dataset
    from Functions import extract
    from Functions import time_normalize
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import torch
    from Functions import tophase
    from models_zoo_alpha import models
    import random
    
    import params as pm

    parser = ArgumentParser()

    parser.add_argument("-model_path",
                        help="Specify the path of the pre trained model")
    parser.add_argument("-test",
                        help="Specify the path of the testing dataset")
    parser.add_argument("-model",
                        help="Specify the path of model to evaluate")
    parser.add_argument("-output",
                        help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Load test dataset
    if args.test is None:
        sys.exit("Please Specify the path of the testing dataset")
    else:
        path_test_dataset = args.test
        name_test_dataset = os.path.splitext(os.path.split(path_test_dataset)[1])[0]
        print('test dataset: ' + name_test_dataset)
    if args.model is None:
        sys.exit("Please Specify the model to evaluate")
    else:
        name_model = args.model
        print('name_model: ' + name_model)
    if args.model_path is None:
        sys.exit("Please specify the path of the pre trained model")
    else:
        path_model = args.model_path
        print('path_model: ' + path_model)
    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        model_name = os.path.split(path_model)[1].split('_state')[0]
        path_figures = os.path.join(path_output, 'figures', model_name+'-on-'+name_test_dataset)
    print('output path: ' + path_output)
    os.makedirs(path_figures, exist_ok=True)

    # Preprocess START
    # Load the validation dataset
    x, x_freq, length = load_dataset(path_test_dataset)

    Num_receiver = x.shape[1]
    Num_shot = x.shape[0]

    # extract traces
    data_traces = extract(x, length)

    # Covert the Re and Im to Phase


    # Preprocess: Normalization magnitude
    data_traces, time_norm_max_buffer = time_normalize(np.array(data_traces))
    data_test = data_traces
    data_test = torch.FloatTensor(data_test)
    

    # Preprocess END

    # Load the model for evaluation
    model, loss_func, optimizer = models(name_model)
    
    if pm.gpu_number is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(pm.gpu_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.is_available():
        TorchTensor = torch.cuda.FloatTensor
    else:
        TorchTensor = torch.FloatTensor
    print("Using Device: {0}, GPU: {1}".format(device, os.environ.get("CUDA_VISIBLE_DEVICES", None)))
    
    model.load_state_dict(torch.load(path_model))
    model.eval()

    # random select n traces to display
    index_test_traces = []

    num_input = pm.num_input
    num_output = pm.num_output

    m = 3  # !!! configure how many shot gather test results you want to show
    num_receiver = x.shape[1]
    # Plot the shot gather
    fig, axes = plt.subplots(1, m, sharex=True, sharey=True)
    imag_low_pred_data = []
    num_shot_list = []
    for i in range(m):
        # test on the shot gather
        num_shot = random.Random(1997 + i).randint(1, x.shape[0] + 1)
        imag_high = np.zeros((num_receiver, num_input))
        imag_low_pred = np.zeros((num_receiver, num_output))
        for j in range(num_receiver):
            imag_high[j] = x[num_shot][j][num_output:]
            input = TorchTensor(imag_high[j])
            input = input.unsqueeze(0).unsqueeze(0)  # CNN expect (batch,
            # channel, len_trace) shape input
            low_pred = model(input)  # after forward prorogation output tensor
            imag_low_pred[j] = low_pred.squeeze(0).detach().cpu().numpy()  # convert to 1d array

        # transpose and make the horizontal is receiver, vertical is time
        imag_high = np.transpose(imag_high)
        imag_low_pred = np.transpose(imag_low_pred)

        # vmin = np.min(imag_high)
        # vmax = np.max(imag_high)
        max_v = np.percentile(imag_high, 99)
        norm = colors.Normalize(vmin=-max_v, vmax=max_v, clip=True)
        axes[i].imshow(imag_high, aspect='auto', norm=norm, cmap='Greys')
        axes[i].set_title('Shot '+str(num_shot))
        if i == 0:
            axes[i].set_xlabel('receiver')
            axes[i].set_ylabel('time')
        imag_low_pred_data.append(imag_low_pred)
        num_shot_list.append(num_shot)
    title = 'Data_10-50Hz'
    fig.suptitle(title, verticalalignment='center')
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close(fig)
    plt.cla()
    
    fig, axes = plt.subplots(1, m, sharex=True, sharey=True)
    for i in range(m):
        num_shot = num_shot_list[i]
        imag_low_pred = imag_low_pred_data[i]
        max_v = np.percentile(np.abs(imag_low_pred), 99)
        norm = colors.Normalize(vmin=-max_v, vmax=max_v, clip=True)
        axes[i].imshow(imag_low_pred, aspect='auto', norm=norm, cmap='Greys')
        axes[i].set_title('Shot '+str(num_shot))
        if i == 0:
            axes[i].set_xlabel('receiver')
            axes[i].set_ylabel('time')
    title = 'Data_Pred_BWE'
    fig.suptitle(title, verticalalignment='center')
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close(fig)
    plt.cla()
    
    del imag_low_pred_data

    # record 3Hz, and 5Hz ground truth and prediction results as image
    img_3hz_gt = []
    img_5hz_gt = []
    img_7hz_gt = []
    img_3hz_predict = []
    img_5hz_predict = []
    img_7hz_predict = []

    num_input_complex = pm.num_input_complex
    num_output_complex = pm.num_output_complex
    num_output_phase = int(num_output_complex / 2)
    num_input_phase = int(num_input_complex / 2)
    num_phase = num_output_phase + num_input_phase
    # calculate all Mean Absolute Errors for whole test dataset, create list to record
    data_traces = np.array(data_traces)
    n = data_traces.shape[0]
    mean_error_trace = []
    error_bars = np.zeros(num_output_phase)
    import time

    for i in range(n):
        # test on the trace
        num_trace = i
        # record all the index of traces which be used to test
        index_test_traces.append(num_trace)

        test_input = data_test.data[num_trace][num_output: data_test.shape[1]]
        # if the new dataset have less input frequency point, then do zero pooling to fit the model's input size.
        test_input_after_zero_pooling = torch.cat([test_input, torch.zeros([num_input - test_input.shape[0]])])

        # evaluate the prediction by trace in frequency domain
        test_input_after_zero_pooling = test_input_after_zero_pooling.unsqueeze(0).unsqueeze(0)
        test_input_after_zero_pooling = test_input_after_zero_pooling.to(device)
        predicted_reconstructed_signal = model(test_input_after_zero_pooling)  # after forward prorogation output tensor
        predicted_reconstructed_signal = predicted_reconstructed_signal.squeeze(0).detach().cpu().numpy()
        time_ground_truth = data_test.data[num_trace][0: num_output]
        time_ground_truth = time_ground_truth.detach().cpu().numpy()

        # record model's prediction and the ground truth phase for each trace
        num_input_complex = pm.num_input_complex
        num_output_complex = pm.num_output_complex
        time_step = pm.time_step  # sample 2 ms
        #time_range = 6  # signal length 5s at the time domin
        time_vec = np.arange(0, time_step*num_input, time_step)
        sig_predicted = predicted_reconstructed_signal
        # FFT only extract the 1-10 Hz frequency components to calculate the total error
        from scipy import fftpack
        sig_true_fft = fftpack.fft(time_ground_truth)[5:51]
        sig_predicted_fft = fftpack.fft(sig_predicted)[5:51]
        # Complex value to Real and Imaginary Part
        complex_sig_fft_trace = []
        complex_sig_predicted_fft_trace = []
        for k in range(sig_true_fft.shape[0]):
            complex_sig_fft_trace = np.append(complex_sig_fft_trace, [sig_true_fft[k].real])
            complex_sig_fft_trace = np.append(complex_sig_fft_trace, [sig_true_fft[k].imag])
            complex_sig_predicted_fft_trace = np.append(complex_sig_predicted_fft_trace, [sig_predicted_fft[k].real])
            complex_sig_predicted_fft_trace = np.append(complex_sig_predicted_fft_trace, [sig_predicted_fft[k].imag])
        # reshape
        complex_sig_fft_trace = complex_sig_fft_trace.reshape(num_output_complex)
        complex_sig_predicted_fft_trace = complex_sig_predicted_fft_trace.reshape(num_output_complex)
        # real, imaginary part to phase
        sig_phase = tophase(complex_sig_fft_trace)  # the size become half
        predicted_phase = tophase(complex_sig_predicted_fft_trace)  # the size become half

        num_output_phase = int(num_output_complex / 2)

        # calculate normalized phase abs for each trace
        y_axis = np.concatenate((sig_phase, np.zeros(int(num_output_phase - sig_phase.shape[0]))), axis=0)
        y_axis_pre = np.concatenate((predicted_phase, np.zeros(num_output_phase - predicted_phase.shape[0])), axis=0)

        # remember the 3,5,7hz
        img_3hz_gt.append(y_axis[12])
        img_3hz_predict.append(y_axis_pre[12])
        img_5hz_gt.append(y_axis[24])
        img_5hz_predict.append(y_axis_pre[24])
        img_7hz_gt.append(y_axis[36])
        img_7hz_predict.append(y_axis_pre[36])

        # ! remember phase wrapping problem use min((error),(error+2pi),(error-2pi))
        abs_error = abs(y_axis_pre[0:num_output_phase] - y_axis[0:num_output_phase])
        abs_error_plus2pi = abs(y_axis_pre[0:num_output_phase] - y_axis[0:num_output_phase] + 2 * np.pi)
        abs_error_minus2pi = abs(y_axis_pre[0:num_output_phase] - y_axis[0:num_output_phase] - 2 * np.pi)
        error_buffer = np.concatenate((abs_error, abs_error_plus2pi, abs_error_minus2pi), axis=0)
        error_buffer = error_buffer.reshape(3, abs_error.shape[0])

        error_bars = error_bars + error_buffer.min(axis=0)  # error is the min((error),(error+2pi),(error-2pi))

        # record abs error for each trace
        mean_error_trace.append(error_buffer.min(axis=0).mean())  # one value for this one trace

    # 3,5,7 Hz shot-receiver frequency slice
    img_3hz_gt = np.array(img_3hz_gt).reshape(Num_shot, Num_receiver)
    img_5hz_gt = np.array(img_5hz_gt).reshape(Num_shot, Num_receiver)
    img_7hz_gt = np.array(img_7hz_gt).reshape(Num_shot, Num_receiver)
    img_3hz_predict = np.array(img_3hz_predict).reshape(Num_shot, Num_receiver)
    img_5hz_predict = np.array(img_5hz_predict).reshape(Num_shot, Num_receiver)
    img_7hz_predict = np.array(img_7hz_predict).reshape(Num_shot, Num_receiver)
    img_3hz_residual = np.min([abs(img_3hz_predict - img_3hz_gt), abs(img_3hz_predict - img_3hz_gt + 2*np.pi),
                               abs(img_3hz_predict - img_3hz_gt - 2*np.pi)], axis=0)
    img_5hz_residual = np.min([abs(img_5hz_predict - img_5hz_gt), abs(img_5hz_predict - img_5hz_gt + 2*np.pi),
                               abs(img_5hz_predict - img_5hz_gt - 2*np.pi)], axis=0)
    img_7hz_residual = np.min([abs(img_7hz_predict - img_7hz_gt), abs(img_7hz_predict - img_7hz_gt + 2*np.pi),
                               abs(img_7hz_predict - img_7hz_gt - 2*np.pi)], axis=0)

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    vmin = np.min(img_3hz_gt)
    vmax = np.max(img_3hz_gt)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_gt = axes[0].imshow(img_3hz_gt, aspect='auto', norm=norm)
    fig.colorbar(im_gt, ax=axes[0])
    axes[0].set_title('3 Hz phase ground truth')
    axes[0].set_xlabel('receiver')
    axes[0].set_ylabel('shot')
    vmin = np.min(img_3hz_predict)
    vmax = np.max(img_3hz_predict)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_predict = axes[1].imshow(img_3hz_predict, aspect='auto', norm=norm)
    fig.colorbar(im_predict, ax=axes[1])
    axes[1].set_title('prediction')
    axes[1].set_xlabel('receiver')
    axes[1].set_ylabel('shot')
    vmin = np.min(img_3hz_residual)
    vmax = np.max(img_3hz_residual)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_residual = axes[2].imshow(img_3hz_residual, aspect='auto', norm=norm)
    fig.colorbar(im_residual, ax=axes[2])
    axes[2].set_title('residual')
    axes[2].set_xlabel('receiver')
    axes[2].set_ylabel('shot')
    title = '3Hz_phase'
    plt.gcf().set_size_inches([9.0, 5.5])
    plt.tight_layout()
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close(fig)
    plt.cla()
    # 5 Hz
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    vmin = np.min(img_5hz_gt)
    vmax = np.max(img_5hz_gt)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_gt = axes[0].imshow(img_5hz_gt, aspect='auto', norm=norm)
    fig.colorbar(im_gt, ax=axes[0])
    axes[0].set_title('5 Hz phase ground truth')
    axes[0].set_xlabel('receiver')
    axes[0].set_ylabel('shot')
    vmin = np.min(img_5hz_predict)
    vmax = np.max(img_5hz_predict)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_predict = axes[1].imshow(img_5hz_predict, aspect='auto', norm=norm)
    fig.colorbar(im_predict, ax=axes[1])
    axes[1].set_title('prediction')
    axes[1].set_xlabel('receiver')
    axes[1].set_ylabel('shot')
    vmin = np.min(img_5hz_residual)
    vmax = np.max(img_5hz_residual)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_residual = axes[2].imshow(img_5hz_residual, aspect='auto', norm=norm)
    fig.colorbar(im_residual, ax=axes[2])
    axes[2].set_title('residual')
    axes[2].set_xlabel('receiver')
    axes[2].set_ylabel('shot')
    title = '5Hz_phase'
    plt.gcf().set_size_inches([9.0, 5.5])
    plt.tight_layout()
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close(fig)
    plt.cla()
    # 7Hz
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    vmin = np.min(img_7hz_gt)
    vmax = np.max(img_7hz_gt)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_gt = axes[0].imshow(img_7hz_gt, aspect='auto', norm=norm)
    fig.colorbar(im_gt, ax=axes[0])
    axes[0].set_title('7 Hz phase ground truth')
    axes[0].set_xlabel('receiver')
    axes[0].set_ylabel('shot')
    vmin = np.min(img_7hz_predict)
    vmax = np.max(img_7hz_predict)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_predict = axes[1].imshow(img_7hz_predict, aspect='auto', norm=norm)
    fig.colorbar(im_predict, ax=axes[1])
    axes[1].set_title('prediction')
    axes[1].set_xlabel('receiver')
    axes[1].set_ylabel('shot')
    vmin = np.min(img_7hz_residual)
    vmax = np.max(img_7hz_residual)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    im_residual = axes[2].imshow(img_7hz_residual, aspect='auto', norm=norm)
    fig.colorbar(im_residual, ax=axes[2])
    axes[2].set_title('residual')
    axes[2].set_xlabel('receiver')
    axes[2].set_ylabel('shot')
    title = '7Hz_phase'
    plt.gcf().set_size_inches([9.0, 5.5])
    plt.tight_layout()
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close(fig)
    plt.cla()

if __name__ == "__main__":
    main()
