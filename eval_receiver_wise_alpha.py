"""eval_receiver_wise_alpha.py: File to evaluate the model's prediction results by receiver-wise output an hdf5 file
to store the prediction result the same size to the test dataset hdf5 file"""

# Example Usage:  python eval_receiver_wise_alpha.py -model /homelocal/AGT_FWI_2020/output/models/CNN11_ResNet_Merged_Marine_BPstatics94_Merged_Marine_BP1997_Marine_Marmousi_state_dict.pt -dataset /homelocal/AGT_FWI_2020/NewDataset/Marine_BP2004_Test.h5 -output /homelocal/AGT_FWI_2020/output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "2.0.0"

# monitor the time for each experiment
import time

start_time = time.time()


def main():
    """ The main function that parses input arguments, read the appropriate
     dataset and model to perform prediction evaluation"""
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import h5py
    from Functions import load_dataset
    from Functions import extract
    from Functions import feature_normalize
    import numpy as np
    import torch
    from models_zoo_alpha import models

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
        name_model = path_model.split('/')[-1].split('_state_')[0]
        print('name_model: ' + name_model)

    # Choose dataset
    if args.dataset is None:
        sys.exit("Specify the path of the dataset to predict")
    else:
        path_dataset = args.dataset
        name_dataset = path_dataset.split('/')[-1].split('.')[0]
        print('name_dataset: ' + name_dataset)

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_output_dataset = path_output + 'datasets/'
    # Parse input arguments END

    # read the dataset to predict
    x, x_freq, length = load_dataset(path_dataset)

    # extract traces
    data_traces = extract(x, length)
    data_traces = feature_normalize(np.array(data_traces))
    data_test = torch.tensor(data_traces).type('torch.FloatTensor')

    # Load the model for evaluation
    num_input_complex = 300
    num_output_complex = 92
    num_receiver = x.shape[1]
    num_shot = x.shape[0]
    x_test_result = np.zeros(x.shape)
    model, loss_func, optimizer = models(name_model.split('_')[0] + '_' + name_model.split('_')[1])
    model.load_state_dict(torch.load(path_model))
    model.eval()
    for i in range(num_shot):
        for j in range(num_receiver):
            num_trace = i * num_receiver + j
            test_input = data_test.data[num_trace][num_output_complex: data_test.shape[1]].unsqueeze(0).unsqueeze(0)
            x_test_result[i][j][:] = np.concatenate(
                (model(test_input).squeeze(0).detach().numpy(), x[i][j][num_output_complex:]),
                axis=0)  # after forward prorogation output tensor

    path = path_output_dataset + name_model + '_Predict_' + name_dataset + '_Result.h5'
    # write the train dataset
    f = h5py.File(path, 'w')
    f.create_dataset('X', data=x_test_result)
    f.create_dataset('X_freq', data=x_freq)
    f.create_dataset('len', data=length)
    f.close()


if __name__ == "__main__":
    main()
