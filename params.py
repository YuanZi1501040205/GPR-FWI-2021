'''
Parameters of this project.
This file is used for storing the parameters that would
not be changed frequently.
'''

# Lengths of input and output traces
num_input = 4241
num_output = 4241

# Batch size
batch_size = 16

# Zoom range, should be in the range of (1, num_output)
zoom_range = (1700, 4241)

# Input and output frequency number
num_input_complex = 25 * 2
num_output_complex = 5 * 2

# The index of 10~300 MHz
freq_range = (1, 31)

# The index of 10-50 MHz errors from 1-40Hz
freq_range_pre = (1, 6)

# Time step
time_step = 2.357934449422306e-11

# GPU number, if no GPU or only one GPU, should use None instead of a number.
gpu_number = None
