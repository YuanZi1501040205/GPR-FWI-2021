"""Time_train_alpha.py: File to train the Neural Networks for GPR_FWI_PROJECT2021 Time Domain Method"""

# python Time_train_alpha.py -train /homelocal/AGT_FWI_2020_Alpha/output/datasets/Time_Marine_Viking_Train.h5 -test /homelocal/AGT_FWI_2020_Alpha/output/datasets/Time_Marine_Viking_Test.h5 -model CNN19_ResUNet -output /homelocal/AGT_FWI_2020_Alpha/output/
# python Time_train_alpha.py -train ./output/datasets/Time_cross_well_geo_model_1_Train.h5 -test ./output/datasets/Time_cross_well_geo_model_1_Test.h5 -model CNN1_ResUNet1 -output ./output/
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# monitor the time for each experiment
import time

start_time = time.time()


def main():
    """ The main function that parses input arguments, calls the appropriate
     Neural Networks models and chose train and test dataset' paths and configure
     the output path. Output 10 Predicted Traces figures and one loss monitor figure, and one evaluation
     results figure at the output path folder"""
    import os
    from argparse import ArgumentParser
    import sys
    import numpy as np
    import torch
    from models_zoo_alpha import models
    from Functions import load_dataset
    from Functions import extract
    from Functions import time_normalize
    import params as pm

    # Parse input arguments START
    parser = ArgumentParser()

    parser.add_argument("-train", help="specify the path of the training dataset")
    parser.add_argument("-test", help="specify the path of the test dataset")
    parser.add_argument("-model", help="Specify the model to train")
    parser.add_argument("-output", help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Choose training dataset
    if args.train is None:
        sys.exit("specify the path of the training dataset")
    else:
        path_train_dataset = args.train
        name_train_dataset = os.path.splitext(os.path.split(path_train_dataset)[1])[0]
        print('training dataset: ' + name_train_dataset)

    # Choose test dataset
    if args.test is None:
        sys.exit("specify the path of the test dataset")
    else:
        path_test_dataset = args.test
        name_test_dataset = os.path.splitext(os.path.split(path_test_dataset)[1])[0]
        print('test dataset: ' + name_test_dataset)

    # Load model
    if args.model is None:
        sys.exit("specify model for training (choose from the models.py)")
    else:
        name_model = args.model

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_figures = os.path.join(path_output, 'figures')
        path_models = os.path.join(path_output, 'models')
    print('output path: ' + path_output)
    os.makedirs(path_models, exist_ok=True)
    os.makedirs(path_figures, exist_ok=True)

    # Choose model from the models.py file
    model, loss_func, optimizer = models(name_model)
    print('model: ' + name_model)

    # assign GPU
    if pm.gpu_number is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(pm.gpu_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Using Device: {0}, GPU: {1}".format(device, os.environ.get("CUDA_VISIBLE_DEVICES", None)))
    # Parse input arguments END

    # Preprocess START
    # Load the training and validation dataset
    x, x_freq, length = load_dataset(path_train_dataset)
    test_x, test_x_freq, test_length = load_dataset(path_test_dataset)

    # extract traces
    data_traces = extract(x, length)
    test_data_traces = extract(test_x, test_length)

    # Shuffle
    import random
    data_traces_shuffled = data_traces[:]
    random.Random(4).shuffle(data_traces_shuffled)  # !!! Random seed can be tuned
    # sample test dataset
    test_data_traces = random.sample(test_data_traces, 50)

    # Create Tensors to hold inputs and outputs
    from torch.autograd import Variable
    # time domain normalization: For each trace, signal / signal.max. Store signal.max for predicted results restore
    data_traces, data_traces_norm_constant = time_normalize(np.array(data_traces_shuffled))
    test_data_traces, test_data_traces_norm_constant = time_normalize(np.array(test_data_traces))
    # convert numpy array to tensor
    data_traces = torch.tensor(data_traces).type('torch.FloatTensor')
    test_data_traces = torch.tensor(test_data_traces).type('torch.FloatTensor')
    data_train = data_traces
    data_test = test_data_traces
    # Preprocess END

    # Set the input output size to build the training dataset
    num_input = pm.num_input
    num_output = pm.num_output

    # Load Dataset as batch
    batch_size = pm.batch_size
    from torch.utils.data import Dataset, DataLoader
    class TrainDataset(Dataset):
        def __init__(self, data_train):
            input_indices = torch.LongTensor(list(np.array(range(num_input)) + num_output))
            output_indices = torch.LongTensor(list(np.array(range(num_output))))

            self.x = torch.index_select(data_train, 1, input_indices)
            self.y = torch.index_select(data_train, 1, output_indices)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_ds = TrainDataset(data_train)
    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    print('Have gotten dataset.')

    # Train
    loss_fig = [[], [],
                [],
                []]  # create loss_fig to store train and validation loss during the epoch (epoch, train_loss, val_loss)
    for epoch in range(1, 81):  # run the model for 120 epochs, epoch can be tuned
        train_loss, valid_loss, test_loss = [], [], []
        # training part
        model.train()
        print('Epoch', epoch)
        # for data in data_train.data:
        for i, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            x = x.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)
            
            # 1. forward propagation
            y_pred = model(x)

            y_pred = y_pred.unsqueeze(1)

            # 2. loss calculation
            loss = loss_func(y_pred, y)

            # 3. backward propagation
            loss.backward()

            # 4. weight optimization
            optimizer.step()

            train_loss.append(loss.item())
            
            if i % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:g}'.format(
                    epoch, i, len(train_dl),
                    100. * i / len(train_dl),
                    loss.item()))

        # print the loss function to monitor the converge
        print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss))

        # Adapt tuning lr
        # # if after 50 epoch the validation loss stop descent, then decrease the learning ratio by divided 10
        # if epoch > 50:
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=1, patience=3,
        #                                                            factor=0.1)
        #     scheduler.step(np.mean(valid_loss))

        # test
        N_test = len(data_test.data)
        for i, data in enumerate(data_test.data):
            optimizer.zero_grad()
            x = data[num_output:num_input + num_output]
            y = data[0:num_output]
            # The cnn only receive the 1*1*num_points as the input tensor size
            x = x.unsqueeze(0).unsqueeze(0).to(device)
            y = y.unsqueeze(0).unsqueeze(0).to(device)

            # 1. forward propagation
            y_pred = model(x)

            y_pred = y_pred.unsqueeze(0)

            # 2. loss calculation
            loss = loss_func(y_pred, y)

            test_loss.append(loss.item())
            
            if i % 5 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:g}'.format(
                    epoch, i, N_test,
                    100. * i / N_test,
                    loss.item()))
        # record loss for each epoch
        loss_fig[0].append(epoch)
        loss_fig[1].append(np.mean(train_loss))
        # loss_fig[2].append(np.mean(valid_loss))
        loss_fig[3].append(np.mean(test_loss))

    # save the model to the output file for reload

    torch.save(model.state_dict(), os.path.join(path_models, name_model + '_' + name_train_dataset + '_state_dict.pt'))

    # save the loss monitor figures
    import matplotlib.pyplot as plt
    with plt.style.context(['science', 'ieee', 'no-latex']):
        fig, ax = plt.subplots()
        plt.plot(loss_fig[0], loss_fig[1], label='Loss of train ' + name_train_dataset)
        # plt.plot(loss_fig[0], loss_fig[2], label='Loss of validation ' + name_val_dataset)
        plt.plot(loss_fig[0], loss_fig[3], label='Loss of test ' + name_test_dataset)
        title = 'Loss of ' + name_model + ' trained on ' + name_train_dataset
        plt.title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
        ax.set(xlabel='Epoch')
        ax.set(ylabel='Loss')
        ax.autoscale(tight=True)
        # fig.savefig('figures/fig1.pdf')
        fig.savefig(os.path.join(path_figures, title + '.png'), dpi=300)
        plt.cla()
    print('done plot')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
