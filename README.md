# GPR-FWI-2020 Alpha

### Current update reports

* 02/03/2021: create repo

### Introduction
This repo contains three groups of .py files which using three strategies to predict the low frequency.
And the Functions.py file contains all self define functions for all these three methods.
The MergeDataset.py also serve both time Domain and Frequency domain methods.

The current dataset size this repo is dealing with viking AVO dataset:

Time domain: 4ms sampling 6s signal for each trace 1500 points

Frequency domain: 
Supervised method:

Frequency domain method: dataset_generator_alpha.py train_alpha.py eval.py eval_receiver_wise_alpha.py

Time domain method: time_dataset_generator_alpha.py Time_train_alpha.py Time_eval_alpha.py Time_eval_receiver_wise_alpha.py

Self supervised method:

Time domain: Self_supervised_train.py
### Download the GitHub repository
```sh
$ cd ~ # local path where you want to download this project
$ git clone https://github.com/YuanZi1501040205/AGT-FWI-2020-Alpha.git
$ cd AGT-FWI-2020-Alpha
```

For environments configure...python3.6

Creat output folder to store the results generated during training and testing.

```sh
$ mkdir -p output/logs output/figures output/models
$ pip install requirements.txt 
```

### Create datasets
The original data for each dataset are two BIN files which is a sequence of time domain data; num_shot*num_receiver numbers 4 bits float. Use the dataset_generator.py or Time_dataset_generator.py to generate the hdf5 dataset and splt it to train and dataset by choosing even number shots(2,4,6,...) as the training and the odd number shots(1,3,5, ...) as the test dataset.
```sh
$ python Time_dataset_generator_alpha.py -split True -path_data /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/Bin -path_label /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/BWE10TO50HZ.Bin -output /homelocal/AGT_FWI_2020_Alpha/output/
```


### Train


Open terminal from the path of project /AGT-FWI-2020-Alpha/
Modify the paths of train dataset and the output folder to yours then run command follows to train:

```sh
$ Time_train_alpha.py -train /homelocal/AGT_FWI_2020_Alpha/output/datasets/Time_Marine_Viking_Train.h5 -test /homelocal/AGT_FWI_2020_Alpha/output/datasets/Time_Marine_Viking_Test.h5 -model CNN19_ResUNet -output /homelocal/AGT_FWI_2020_Alpha/output/
```


### Check the output/figure folder

Get the loss figure during training? Great! You have successfully trained the model.
### Test

Let's evaluated the model we just trained. Open terminal from the path of project /AGT-FWI-2020-Alpha/
Modify the command below then run :

```sh
$ python Time_eval_alpha.py -test /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/time/Time_Marine_Viking_Test.h5 -model CNN19_ResUNet1 -model_path /homelocal/AGT_FWI_2020_Alpha/output/models/CNN19_ResUNet1_Time_Marine_Viking_Train_state_dict.pt -output /homelocal/AGT_FWI_2020_Alpha/output/
```

Check all your training and the evaluation/test result from the output/figures folder.

Finally evaluate the pre-trained model by using eval_receiver_wise.py:
```sh
$ python Time_eval_receiver_wise_alpha.py -model /homelocal/AGT_FWI_2020_Alpha/output/models/CNN19_ResUNet1_Time_Marine_Viking_Train_state_dict.pt -dataset /homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/viking/time/Time_Marine_Viking_Test.h5 -output /homelocal/AGT_FWI_2020_Alpha/output/
```

There should be a predicted results hdf5 file appear at the output/dataset folder. Use the AGT-FWI-2019 display tool to show this hdf5 
file and the ground truth hdf5 file.  

