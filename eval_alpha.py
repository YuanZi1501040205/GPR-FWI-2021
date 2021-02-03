"""eval_alpha.py: File to perform the evaluation for models trained by the train.py file Frequency Domain"""

# Complex CNN Example Usage: python eval_alpha.py -test /homelocal/AGT_FWI_2020/NewDataset/BP2004_Test.h5 -model CNN1_vgg -model_path /homelocal/AGT_FWI_2020/output/models/CNN1_vgg_BP2004_Train_state_dict.pt -output /homelocal/AGT_FWI_2020/output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "3.0.0"


def main():
    """ The main function that parses input arguments, load the appropriate
     Neural Networks models, chose the test dataset' path and configure
     the output path. Randomly Sample n Predicted Traces figures and calculate the statistic
     evaluation results by receiver-wise and frequency-wise shot_wise (developing offset-wise). Plot result figures to
     the output folder"""

    # Parse input arguments
    from argparse import ArgumentParser
    import sys
    from Functions import load_dataset
    from Functions import extract
    from Functions import feature_normalize
    from Functions import mean_phase_abs_error
    import numpy as np

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
        name_test_dataset = path_test_dataset.split("/")[-1].split(".")[0]
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
        path_figures = path_output + 'figures/'

    # Preprocess START
    # Load the validation dataset
    x, x_freq, length = load_dataset(path_test_dataset)

    # extract traces
    data_traces = extract(x, length)

    # There is no shuffle for test procedure

    import torch
    # Magnitude Normalization, 1Hz_real,1Hz_image=>1Hz_real/sqrt(1Hz_real^2 + 1Hz_image^2), 1Hz_image/sqrt
    # (1Hz_real^2 + 1Hz_image^2)
    data_traces = feature_normalize(np.array(data_traces))
    data_test = data_traces

    # numpy array to tensor
    data_test = torch.tensor(data_test).type('torch.FloatTensor')

    # Preprocess END

    # Load the model for evaluation
    from models_zoo_alpha import models
    model, loss_func, optimizer = models(name_model)
    model.load_state_dict(torch.load(path_model))
    model.eval()

    # random select n traces to display
    import random
    index_test_traces = []

    # set the input and the output size manually for every model chose from models.py file
    num_input = 240*2
    num_output = 55*2

    n = 5  # configure the number of traces' test result to be plotted
    for i in range(n):
        # test on the trace
        num_trace = random.Random(1997 + i).randint(1, data_test.shape[0] + 1)
        # record all the index of traces which be used to test, convert num_shot*num_receiver index to shot ? receiver ?
        shot_index = int(num_trace / x.shape[1]) + 1
        receiver_index = num_trace % x.shape[1]
        index_test_traces.append(num_trace)


        test_input = data_test.data[num_trace][num_output: data_test.shape[1]]
        # if the new dataset have less input frequency point, then do zero pooling to fit the model's input size.
        test_input_after_zero_pooling = torch.cat([test_input, torch.zeros([num_input - test_input.shape[0]])])

        if "CNN" in name_model:  # if the model is CNN or RNN(developing)
            num_input_complex = 240*2
            num_output_complex = 55*2

            # The CNN expect to receive the (batch_size, channel, num_points) as the input tensor size.
            test_input_after_zero_pooling = test_input_after_zero_pooling.unsqueeze(0).unsqueeze(0)

        from Functions import tophase
        predicted_low_freq_complex = model(test_input_after_zero_pooling)  # forward prorogation
        predicted_low_freq_complex = predicted_low_freq_complex.reshape(num_output_complex)
        predicted_low_freq = tophase(predicted_low_freq_complex)  # Real and image parts => phase. the size become half
        ground_truth = tophase(data_test[num_trace])  # phase ground truth
        num_output_phase = int(num_output_complex / 2)
        num_input_phase = int(num_input_complex / 2)
        num_phase = num_output_phase + num_input_phase

        # place_holders to store ground truth and predicted results
        real_phase = np.zeros(int(ground_truth.shape[0]))
        predicted_low_freq_phase = np.zeros(int(ground_truth.shape[0]))

        # fill the real_phase and the predicted_phase with ground truth and the predicted points
        for j in range(ground_truth.shape[0]):

            # use predicted points(Real and Imaginary pairs) to calculate predicted phase points
            if j < num_output_phase:
                real_phase[j] = ground_truth[j]
                predicted_low_freq_phase[j] = predicted_low_freq[j]

            elif j >= num_output_phase:
                real_phase[j] = ground_truth[j]
                predicted_low_freq_phase[j] = ground_truth[j]

        # prepare results for plot
        import matplotlib.pyplot as plt
        with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
            fig, ax = plt.subplots()
            x_axis = list(np.arange(1, 10 + 1/6, 1/6))
            y_axis = np.concatenate((real_phase, np.zeros(int(num_phase - real_phase.shape[0]))), axis=0)
            plt.plot(x_axis[0:55], list(y_axis)[0:55], label='Ground Truth')  # plot the line of ground truth
            # ONLY plot the predicted points 1-10Hz 0:46 points
            y_axis_pre = np.concatenate(
                (predicted_low_freq_phase, np.zeros(num_phase - predicted_low_freq_phase.shape[0])),
                axis=0)
            plt.plot(x_axis[0:55], list(y_axis_pre)[0:55],
                     label='Predicted Phase')  # plot the line of predicted results
            # ONLY plot the predicted points 1-10Hz 0:46 points
            title = path_model.split('/')[-1].split('_state_')[0] + ' tested on Shot.' + str(
                shot_index) + ' Receiver ' + str(receiver_index) + ' trace of ' + name_test_dataset
            # calculate normalized phase MAE for each traces
            mean_error_trace = mean_phase_abs_error(predicted_low_freq_phase[0:num_output_phase],
                                                    real_phase[0:num_output_phase])
            plt.text(0, -5, 'Mean abs error for this trace:' + str(mean_error_trace), style='italic',
                     bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 4})
            error_bars_buffer = [abs(predicted_low_freq_phase - real_phase),
                                 abs(predicted_low_freq_phase - real_phase + 2 * np.pi),
                                 abs(predicted_low_freq_phase - real_phase - 2 * np.pi)]
            error_bars = np.array(error_bars_buffer).min(axis=0)
            # plot error bar at the prediction results figures
            plt.errorbar(x_axis[0:55], y_axis[0:55], yerr=error_bars[0:28],
                         fmt='.k');  # !!! ONLY plot the predicted points 1-10Hz 0:46 points
            plt.title(title)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
            ax.set(xlabel='Frequency (Hz)')
            ax.set(ylabel='Phase (Radians)')
            ax.set(ylim=[-3.3, 3.3])
            ax.set(xlim=[0.8, 10.2])
            fig.savefig(path_figures + title + '.png', dpi=300)
            plt.close(fig)
            plt.cla()
            print("Plotted sample: " + str(i))

    # Count normalized MAE for whole test dataset by frequency-wise, create list to record
    data_traces = np.array(data_traces)
    n = data_traces.shape[0]
    mean_error_trace = []
    error_bars = np.zeros(num_output_phase)  # set sum = 0 for each frequency component
    for i in range(n):
        # test on the trace
        num_trace = i
        # record all the index of traces which be used to test
        index_test_traces.append(num_trace)

        test_input = data_test.data[num_trace][num_output: data_test.shape[1]]
        # if the new dataset have less input frequency point, then do zero pooling to fit the model's input size.
        test_input_after_zero_pooling = torch.cat([test_input, torch.zeros([num_input - test_input.shape[0]])])

        # record model's prediction and the ground truth phase for each trace

        if "CNN" in name_model:  # if the model is CNN or RNN
            num_input_complex = 240
            num_output_complex = 55

            # The CNN expect to receive the (batch_size, channel, num_points) as the input tensor size.
            test_input_after_zero_pooling = test_input_after_zero_pooling.unsqueeze(0).unsqueeze(0)

        predicted_low_freq_complex = model(test_input_after_zero_pooling)
        predicted_low_freq_complex = predicted_low_freq_complex.reshape(
            num_output_complex)  # critical because the shape of CNN output is not one dimension vector
        predicted_low_freq = tophase(predicted_low_freq_complex)  # the size become half
        ground_truth = tophase(data_test[num_trace])  # ground truth is the phase
        num_output_phase = int(num_output_complex / 2)
        num_input_phase = int(num_input_complex / 2)
        num_phase = num_output_phase + num_input_phase

        # place_holders for ground truth and predicted results
        real_phase = np.zeros(int(ground_truth.shape[0]))
        predicted_low_freq_phase = np.zeros(int(ground_truth.shape[0]))

        # fill the real_phase and the predicted_phase with ground truth and the predicted points
        for j in range(ground_truth.shape[0]):

            # use predicted points to calculate predicted phase points
            if j < num_output_phase:
                real_phase[j] = ground_truth[j]
                predicted_low_freq_phase[j] = predicted_low_freq[j]

            elif j >= num_output_phase:
                real_phase[j] = ground_truth[j]
                predicted_low_freq_phase[j] = ground_truth[j]

        # calculate normalized phase MAE for each traces
        # !remember phase wrapping problem use min((error),(error+2pi),(error-2pi))
        abs_error = abs(predicted_low_freq_phase[0:num_output_phase] - real_phase[0:num_output_phase])
        abs_error_plus2pi = abs(
            predicted_low_freq_phase[0:num_output_phase] - real_phase[0:num_output_phase] + 2 * np.pi)
        abs_error_minus2pi = abs(
            predicted_low_freq_phase[0:num_output_phase] - real_phase[0:num_output_phase] - 2 * np.pi)
        error_buffer = np.concatenate((abs_error, abs_error_plus2pi, abs_error_minus2pi), axis=0)
        error_buffer = error_buffer.reshape(3, abs_error.shape[0])

        error_bars = error_bars + error_buffer.min(axis=0)  # error is the min((error),(error+2pi),(error-2pi))

        # record wrapped MAE for each trace. mean_error_trace.shape is (num_shots*num_receivers)
        mean_error_trace.append(error_buffer.min(axis=0).mean())  # one value for this one trace

    # mean of all traces
    error_bars = np.array(error_bars) / n
    total_abs = np.array(error_bars).mean()
    print("Total average abs error: ", total_abs)
    # plot the error bars and the total MAE to the results figure
    with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
        fig, ax = plt.subplots()
        plt.bar(x_axis[0:num_output_phase], error_bars)
        plt.text(0, -0.08, 'Mean abs error:' + str(total_abs), style='italic',
                 bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 3})
        model_name = path_model.split('/')[-1].split('_state')[0]
        title = model_name + ' tested on ' + name_test_dataset + '(Frequency-wise)'
        plt.title(title)
        ax.set(xlabel='Frequency (Hz)')
        ax.set(ylabel='Phase Mean Absolute Errors (Radians)')
        ax.autoscale(tight=True)
        fig.savefig(path_figures + title + '.png', dpi=300)
        plt.close(fig)
        plt.cla()

    # Errors statistic receiver-wise
    # plot the error bars and the total MAE to the results figure
    num_shot = x.shape[0]
    num_receiver = x.shape[1]

    x_axis = range(num_receiver)
    mean_error_trace = np.array(mean_error_trace)  # mean_error_trace.shape is (num_shots*num_receivers)
    # each entry of this vector represent the statistic of this trace's error
    mean_error_trace = mean_error_trace.reshape(num_shot, num_receiver)
    y_axis = mean_error_trace.mean(axis=0)
    with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
        fig, ax = plt.subplots()
        plt.bar(x_axis, y_axis)
        plt.text(0, -0.15, 'Mean abs error:' + str(total_abs), style='italic',
                 bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 4})
        model_name = path_model.split('/')[-1].split('_state')[0]
        title = model_name + ' tested on' + name_test_dataset + '(Receiver-wise)'
        plt.title(title)
        ax.set(xlabel='Receiver')
        ax.set(ylabel='Phase Mean Absolute Errors (Radians)')
        ax.autoscale(tight=True)
        fig.savefig(path_figures + title + '.png', dpi=300)
        plt.close(fig)
        plt.cla()

    # Shot-wise predicted results' statistic
    x_axis = range(num_shot)
    # mean_error_trace.shape is (num_shots, num_receivers)
    # each entry of this vector represent the statistic of this trace's error
    y_axis = mean_error_trace.mean(axis=1)
    with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
        fig, ax = plt.subplots()
        plt.bar(x_axis, y_axis)
        plt.text(0, -0.15, 'Mean abs error:' + str(total_abs), style='italic',
                 bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 4})
        model_name = path_model.split('/')[-1].split('_state')[0]
        title = model_name + ' tested on' + name_test_dataset + '(Shot-wise)'
        plt.title(title)
        ax.set(xlabel='Shot')
        ax.set(ylabel='Phase Mean Absolute Errors (Radians)')
        ax.autoscale(tight=True)
        fig.savefig(path_figures + title + '.png', dpi=300)
        plt.close(fig)
        plt.cla()

    print("done")


if __name__ == "__main__":
    main()
