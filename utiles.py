# %% prepare pst file
num_shot = 13
num_receiver = 13
length_sig = 4241
len_x = 800
len_y = 800
num_obs = num_shot*num_receiver*length_sig
num_para = len_x * len_y
file1 = open("/home/yzi/research/PEST/pest/pest_source/gpr/inversion_p.pst", "w")
ctrl_data_lines = ["pcf\n", "* control data\n", "restart  estimation\n", "    640000    716729     640000     0     1\n"
    , "    1     1 single point   1   0   0\n",  "  5.0   2.0   0.3  0.03    10\n",
     "  3.0   3.0 0.001\n", "  0.1\n   20  0.01     3     3  0.01     3\n", "    0     0     0\n"]
# parameter groups
# initial parameter data
para_lines = ["* parameter groups\n"]
para_data_lines = ["* parameter data\n"]
for i in range(len_x):
    for j in range(len_y):
        name_para = "p" + str(i) + "_" + str(j)
        para_lines.append(name_para + "           relative 0.01  0.0  switch  2.0 parabolic\n")
        para_data_lines.append(name_para + "           none relative   5.500000      -1.000000E+10   1.000000E+10 "
        +name_para+"              1.0000        0.0000      1\n")
# %%
obs = X
# %% real observation data
obs_groups_lines = ["* observation groups\nobsgroup\n"]
obs_data_lines = ["* observation data\n"]
for i in range(num_shot):
    for j in range(num_receiver):
        for k in range(length_sig):
            name_obs = "o" + str(i) + "_" + str(j) + "_" + str(k)
            obs_f = ("%.6f"%obs[i][j][k])
            obs_data_lines.append(name_obs + "            "+str(obs_f)+"      1.0  obsgroup\n")
# %% model control files
model_control_lines = ["* model command line\n", "./forward.sh\n", "* model input/output\n",
                       "in.tpl  in.dat\n", "out.ins out.dat\n", "* prior information\n"]
# \n is placed to indicate EOL (End of Line)
file1.writelines(ctrl_data_lines)
file1.writelines(para_lines)
file1.writelines(para_data_lines)
file1.writelines(obs_groups_lines)
file1.writelines(obs_data_lines)
file1.writelines(model_control_lines)
file1.close()  # to change file access modes
# %%
import os
os.system('export PATH=$PATH:/home/yzi/research/PEST/pest/pest_source/')
# %% make input template
file = open("/home/yzi/research/PEST/pest/pest_source/gpr/in.tpl", "w")
in_tpl_headlines = ["ptf #\n"]
for i in range(len_x):
    for j in range(len_y):
        name_para = "p" + str(i) + "_" + str(j)
        in_tpl_headlines.append("# " + name_para + "        #\n")
file.writelines(in_tpl_headlines)
file.close()  # to change file access modes
# %% make input data initial parameters(permittivity)
file = open("/home/yzi/research/PEST/pest/pest_source/gpr/in.dat", "w")
permittivity_initial = 5.5
in_data_headlines = []
permittivity_initial=("%e"%permittivity_initial)
for i in range(len_x):
    for j in range(len_y):
        name_para = "p" + str(i) + "_" + str(j)
        in_data_headlines.append(str(permittivity_initial) + "\n")
file.writelines(in_data_headlines)
file.close()  # to change file access modes
# %% output data observation during training
# %% current observation data for out.dat
file = open("/home/yzi/research/PEST/pest/pest_source/gpr/out.dat", "w")
# input current parameter=>forward=> current observation
cur_obs_data_lines = [""]
for i in range(num_shot):
    for j in range(num_receiver):
        for k in range(length_sig):
            formator = ("%e"%obs[i][j][k])
            cur_obs_data_lines.append(str(formator)+"\n")
file.writelines(in_data_headlines)
file.close()  # to change file access modes
# %% observation data instruction for data.ins
file = open("/home/yzi/research/PEST/pest/pest_source/gpr/out.ins", "w")
# input current parameter=>forward=> current observation
out_ins_lines = ["pif #\n"]
for i in range(num_shot):
    for j in range(num_receiver):
        for k in range(length_sig):
            name_obs = "o" + str(i) + "_" + str(j) + "_" + str(k)
            out_ins_lines.append("l1 ("+name_obs+ ")1:12")
file.writelines(out_ins_lines)
file.close()  # to change file access modes
# %% convert output of gprmax to obsedrvation matrix obs[shot][receiver][length of signal]
import h5py
import numpy as np
X = np.zeros((num_shot, num_receiver, length_sig))
path = '/home/yzi/research/GPR-FWI-2021/geo_model/hdf5/cross_well_cylinder_B_scan_shot_'
for i in range(num_shot):
    path_f = path + str(i) + '.out'
    h5_file = h5py.File(path_f, 'r')
    rxs = h5_file['rxs']
    for j in range(num_receiver):
        rx = rxs['rx' + str(j + 1)]
        X[i][j][:] = rx['Ez'][()]
# %% generate ground truth's input files for gprmax
folder_path = "/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt/forward_input_gt"
for i in range(num_shot):
    path_file = folder_path + 'cross_well_cylinder_B_scan_shot_' + str(i) + '.in'
    file = open(path_file, "w")
    # input current parameter=>forward=> current observation
    head_lines = ["#title: cross boreholes B-scan from two cylinders buried in ground\n",
                  "#domain: 8.0 8.0 0.01\n",
                  "#dx_dy_dz: 0.01 0.01 0.01\n",
                  "#time_window: 10e-08\n\n"
                  ]
    offset = 0.5
    shot_start = 1.0
    shot_index = i*offset + shot_start
    shot_receiver_line = ["#waveform: ricker 1.0 100000000.0 mysource\n",
                          "#hertzian_dipole: z 1.0 " +str(shot_index)+" 0.0 mysource\n",
                          "#rx_array: 7.0 1.0 0.0 7.0 7.0 0.0 0.0 0.5 0.0\n\n"]

    material_lines = ["#material: 5.5 0.0028 1.0 0.0 mid\n",
                   "#material: 5.0 0.001 1.0 0.0 top\n",
                   "#material: 6.5 0.008 1.0 0.0 pipe\n",
                   "#material: 5.0 0.001 1.0 0.0 bg\n\n"]

    geo_model_lines = ["#box: 0.0 0.0 0.0 8.0 8.0 0.01 top\n",
                       "#cylindrical_sector: z 0.0 18.0 0.0 0.01 16.0 270.0 90.0 mid\n",
                       "#box: 0.0 6.0 0.0 8.0 8.0 0.01 top\n",
                       "#cylinder: 3.0 4.0 0.0 3.0 4.0 0.01 0.5 pipe\n",
                       "#cylinder: 5.0 4.0 0.0 5.0 4.0 0.01 0.5 pipe\n",
                       "#geometry_view: 0.0 0.0 0.0 8.0 8.0 0.01 0.01 0.01 0.01 cross_well_half_space n\n"]

    file.writelines(head_lines)
    file.writelines(shot_receiver_line)
    file.writelines(material_lines)
    file.writelines(geo_model_lines)
    file.close()  # to change file access modes
# %% gprmax forward
folder_path = "/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt/forward_input_gt"
import os
os.system('cd /home/yzi/research/gprMax\n'
          +'conda env create -f conda_env.yml\n'
          + 'activate gprMax\n'
        +'python setup.py build\n'
        +'python setup.py install\n'
#+'python -m gprMax /home/yzi/research/GPR-FWI-2021/forward/forward_input_gt/forward_input_gtcross_well_cylinder_B_scan_shot_0.in'
       )


for i in range(num_shot):
    path_file = folder_path + 'cross_well_cylinder_B_scan_shot_' + str(i) + '.in'
    os.system('python -m ' + '/home/yzi/research/gprMax/gprMax '+ path_file)


