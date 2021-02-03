# AGT-FWI-2020 Alpha

*Click [here](./index) to go back to the content.*

In this report, we would show results of two tests. One example of the data is shown as below:

| Time domain | Frequency amplitude |
| :-----: | :-----: |
| ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/dataset.png) | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/dataset_freq.png) |

In this example,

* The first row is the input data, the range of the frequency is about `(10, 35)`.
* The second row is the label data, it is spiky and given by optimization.
* The third row is the low-pass label data. It is produced by `<10Hz` filter performed on the second row.
* The last row is the ricker wavelet. It shows the relationship between the first row and the second row.

There are two datasets used in this report:

`TDATA_DMBP450x160zD10_56S112R_RICK15HZ`: The input is the first row. The label is the second row.
`TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW`: The input is the first row. The label is the third row.

## Test 1

To run this test, use the following codes:

```bash
# Train
python Time_train_alpha.py -train ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ-train.h5 -test ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ-test.h5 -model CNN19_ResUNet1 -output ./output
# Test
python Time_eval_alpha.py -test ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ-test.h5 -model CNN19_ResUNet1 -model_path ./output/models/CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ-train_state_dict.pt -output ./output
# Prediction
python Time_eval_receiver_wise_alpha.py -model ./output/models/CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ-train_state_dict.pt -dataset ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ-test.h5 -output ./output
```

This test is performed on the dataset `TDATA_DMBP450x160zD10_56S112R_RICK15HZ`. To view all results, please check [here](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ).

After training 80 epochs, we could get the training loss:

| Training loss |
| :-----: |
| ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Loss.png) |

We show 4 examples of the predictions here:

| Example | Figure |
| ----- | :-----: |
| Shot 8 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Shot_8.png) |
| Shot 38 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Shot_38.png) |
| Shot 50 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Shot_50.png) |

The phase plot is shown as:

| Freq. | Figure |
| ----- | :-----: |
| 3Hz | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/3Hz_phase.png) |
| 5Hz | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/5Hz_phase.png) |
| 7Hz | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/7Hz_phase.png) |

We could also show some examples as:

| Example | Figure |
| ----- | :-----: |
| Shot 9, Receiver 62 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Zoom_Shot_9_Receiver_62.png) |
| Shot 18, Receiver 25 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Zoom_Shot_18_Receiver_25.png) |
| Shot 43, Receiver 101 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Zoom_Shot_43_Receiver_101.png) |

We could also show some examples of power spectrums:

| Freq. | Figure |
| ----- | :-----: |
| Shot 9, Receiver 62 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Shot_9_Receiver_62_Power.png) |
| Shot 18, Receiver 25 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Shot_18_Receiver_25_Power.png) |
| Shot 43, Receiver 101 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ/Shot_43_Receiver_101_Power.png) |

## Test 2

To run this test, use the following codes:

```bash
# Train
python Time_train_alpha.py -train ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-train.h5 -test ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-test.h5 -model CNN19_ResUNet1 -output ./output
# Test
python Time_eval_alpha.py -test ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-test.h5 -model CNN19_ResUNet1 -model_path ./output/models/CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-train_state_dict.pt -output ./output
# Prediction
python Time_eval_receiver_wise_alpha.py -model ./output/models/CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-train_state_dict.pt -dataset ./datasets/TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW-test.h5 -output ./output
```

This test is performed on the dataset `TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW`. To view all results, please check [here](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW).

After training 80 epochs, we could get the training loss:

| Training loss |
| :-----: |
| ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Loss.png) |

We show 4 examples of the predictions here:

| Example | Figure |
| ----- | :-----: |
| Shot 8 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Shot_8.png) |
| Shot 38 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Shot_38.png) |
| Shot 50 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Shot_50.png) |

The phase plot is shown as:

| Freq. | Figure |
| ----- | :-----: |
| 3Hz | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/3Hz_phase.png) |
| 5Hz | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/5Hz_phase.png) |
| 7Hz | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/7Hz_phase.png) |

We could also show some examples as:

| Example | Figure |
| ----- | :-----: |
| Shot 9, Receiver 62 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Zoom_Shot_9_Receiver_62.png) |
| Shot 18, Receiver 25 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Zoom_Shot_18_Receiver_25.png) |
| Shot 43, Receiver 101 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Zoom_Shot_43_Receiver_101.png) |

We could also show some examples of power spectrums:

| Freq. | Figure |
| ----- | :-----: |
| Shot 9, Receiver 62 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Shot_9_Receiver_62_Power.png) |
| Shot 18, Receiver 25 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Shot_18_Receiver_25_Power.png) |
| Shot 43, Receiver 101 | ![](./CNN19_ResUNet1_TDATA_DMBP450x160zD10_56S112R_RICK15HZ_LOW/Shot_43_Receiver_101_Power.png) |
