"""train_alpha.py: File to train the Neural Networks for AGT_FWI_PROJECT2020"""

#python train_alpha.py -train /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/Frequency/Marine_VikingBWE_Train.h5 -test /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/Frequency/Marine_VikingBWE_Test.h5 -model CNN11_ResNet2 -output /homelocal/AGT_FWI_2020/output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "3.0.0"

# monitor the time for each experiment
import time

start_time = time.time()


def main():
    """ The main function that parses input arguments, calls the appropriate
     Neural Networks models and configure the input and output paths. Plot one loss descent figure at the output folder
     Inputs: Train dataset to train the neural network, validation and test dataset to test if the training works
     Output: Loss figure"""

    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    from models_zoo_alpha import models
    from Functions import load_dataset
    from Functions import extract
    from Functions import feature_normalize

    parser = ArgumentParser()

    parser.add_argument("-train", help="specify the path of the training dataset")
    parser.add_argument("-val", help="specify the path of the validation dataset")
    parser.add_argument("-test", help="specify the path of the test dataset")
    parser.add_argument("-model", help="Specify the model to train")
    parser.add_argument("-output", help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Choose train dataset
    if args.train is None:
        sys.exit("specify the path of the training dataset")
    else:
        path_train_dataset = args.train
        name_train_dataset = path_train_dataset.split('.')[0].split('/')[-1]
        print('training dataset: ' + name_train_dataset)

    # Choose test dataset
    if args.test is None:
        sys.exit("specify the path of the test dataset")
    else:
        path_test_dataset = args.test
        name_test_dataset = path_test_dataset.split('.')[0].split('/')[-1]
        print('test dataset: ' + name_test_dataset)

    # Choose the model
    if args.model is None:
        sys.exit("specify model for training (choose from the models.py)")
    else:
        name_model = args.model

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_figures = path_output + 'figures/'
        path_models = path_output + 'models/'

    # Choose model from the models.py file
    model, loss_func, optimizer = models(name_model)
    print('model: ' + name_model)

    # assign GPU to run the experiment
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change this "int" number to use different GPU:0, 1, 2 or 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Using GPU: " + os.environ["CUDA_VISIBLE_DEVICES"])
    # Parse input arguments END

    # Preprocess START
    # Load the training, validation and test datasets. x.shape is (num_shot, num_receiver, frequency[Re_1Hz, Im_1Hz, ...
    # , Re_50Hz, Im_50Hz]); x_freq is (240 * 2) size vector [1Hz, , 50Hz]; the length.shape is (num_shot) [
    # num_receiver, ..., num_receiver]
    x, x_freq, length = load_dataset(path_train_dataset)
    # val_x, val_x_freq, val_length = load_dataset(path_val_dataset)
    test_x, test_x_freq, test_length = load_dataset(path_test_dataset)

    # extract traces: data_traces.shape is (number of all traces in this dataset, frequency[Re_1Hz, Im_1Hz, ...
    # , Re_50Hz, Im_50Hz])
    data_traces = extract(x, length)
    test_data_traces = extract(test_x, test_length)

    # Shuffle
    import random
    data_traces_shuffled = data_traces[:]
    random.Random(4).shuffle(data_traces_shuffled)  # Random seed can be changed for tuning

    # Create Tensors to hold inputs and outputs
    from torch.autograd import Variable
    # Directly predict complex number(two value for one data point)

    # Magnitude Normalization, 1Hz_real,1Hz_image=>1Hz_real/sqrt(1Hz_real^2 + 1Hz_image^2), 1Hz_image/sqrt
    # (1Hz_real^2 + 1Hz_image^2)
    data_traces = feature_normalize(np.array(data_traces_shuffled))
    # val_data_traces = feature_normalize(np.array(val_data_traces))
    test_data_traces = feature_normalize(np.array(test_data_traces))

    # Convert numpy array to tensor
    data_traces = torch.tensor(data_traces).type('torch.FloatTensor')
    # val_data_traces = torch.tensor(val_data_traces).type('torch.FloatTensor')
    test_data_traces = torch.tensor(test_data_traces).type('torch.FloatTensor')

    # After data preprocess, export three matrix: data_train, data_val and data_test
    data_train = data_traces
    # data_val = val_data_traces
    data_test = test_data_traces
    # Preprocess END

    # patch for viking
    num_input = 240 * 2
    num_output = 55 * 2

    # Load Dataset as batch
    batch_size = 4

    # For each trace, split the 1-10Hz as the output of the neural network and the 10-40Hz as the input of the neural
    # network.
    from torch.utils.data import Dataset, DataLoader

    class TrainDataset(Dataset):
        def __init__(self, dataset):
            input_indices = torch.LongTensor(list(np.array(range(num_input)) + num_output))
            output_indices = torch.LongTensor(list(np.array(range(num_output))))

            self.x = torch.index_select(dataset, 1, input_indices)
            self.y = torch.index_select(dataset, 1, output_indices)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_ds = TrainDataset(data_train)

    # DataLoader data_train.shape is (number of all traces in this dataset, frequency[Re_1Hz, Im_1Hz, ... ,
    # Re_40Hz, Im_40Hz])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    data_train = train_dl

    # Train
    loss_fig = [[], [], []]  # create loss_fig to store train, validation and test loss during the epoch iteration.
    # loss_fig.shape is (epoch, train_loss, val_loss, test_loss)
    for epoch in range(1, 81):  # run the model for 80 epochs. epoch number can be tuned
        train_loss, test_loss = [], []

        # training part, configure the model to training mode
        model.train()
        # for data in data_train.data:
        for i, (x, y) in enumerate(data_train):
            optimizer.zero_grad()
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            # The CNN expect to receive the (batch_size, channel, num_points) as the input tensor size.
            if "CNN" in name_model:
                x = x.unsqueeze(1).to(device)
                y = y.unsqueeze(1).to(device)

            # 1. forward propagation
            y_pred = model(x)

            if "CNN" in name_model:
                y_pred = y_pred.unsqueeze(1)

            # 2. loss calculation
            loss = loss_func(y_pred, y)

            # 3. backward propagation
            loss.backward()

            # 4. weight optimization
            optimizer.step()

            train_loss.append(loss.item())


        # test
        for data in data_test.data:
            optimizer.zero_grad()
            x = Variable(data[num_output:num_input + num_output]).to(device)
            y = Variable(data[0:num_output]).to(device)
            # The CNN expect to receive the (batch_size, channel, num_points) as the input tensor size.
            if "CNN" in name_model:
                x = x.unsqueeze(0).unsqueeze(0).to(device)
                y = y.unsqueeze(0).unsqueeze(0).to(device)

            # 1. forward propagation
            y_pred = model(x)

            if "CNN" in name_model:  # Convert tensor y_pred from (1,96)=>(1,1,96)
                y_pred = y_pred.unsqueeze(0)

            # 2. loss calculation
            loss = loss_func(y_pred, y)

            test_loss.append(loss.item())

        # print the loss function to monitor the converge
        print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss))
        # # if after 50 epoch the validation loss stop descent, then decrease the learning ratio by divided 10
        # if epoch > 50:
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=1, patience=3,
        #                                                            factor=0.1)
        #     scheduler.step(np.mean(valid_loss))

        # record loss for each epoch
        loss_fig[0].append(epoch)
        loss_fig[1].append(np.mean(train_loss))
        # loss_fig[2].append(np.mean(valid_loss))
        loss_fig[2].append(np.mean(test_loss))

    # save the model to the output file for evaluation
    torch.save(model.state_dict(), path_models + name_model + '_' + name_train_dataset + '_state_dict.pt')

    # PLot and save the loss monitor figures
    import matplotlib.pyplot as plt
    with plt.style.context(['science', 'ieee', 'no-latex']):
        fig, ax = plt.subplots()
        plt.plot(loss_fig[0], loss_fig[1], label='Loss of train ' + name_train_dataset)
        plt.plot(loss_fig[0], loss_fig[2], label='Loss of test ' + name_test_dataset)
        title = 'Loss of ' + name_model + ' trained on ' + name_train_dataset
        plt.title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
        ax.set(xlabel='Epoch')
        ax.set(ylabel='Loss')
        ax.autoscale(tight=True)
        fig.savefig(path_figures + title + '.png', dpi=300)
        plt.cla()
    print('done plot')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
