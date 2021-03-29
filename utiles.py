import numpy as np
import os
import sys
import h5py

class tools():
    def __init__(self, num_shot=13, num_receiver=13, length_sig=425, len_x=80, len_y=80):
        self.num_shot = num_shot
        self.num_receiver = num_receiver
        self.length_sig = length_sig
        self.len_x = len_x
        self.len_y = len_y
        self.real_obs = np.zeros((num_shot, num_receiver, length_sig))
        self.curr_obs = np.zeros((num_shot, num_receiver, length_sig))
        self.num_obs = num_shot * num_receiver * length_sig
        self.num_para = len_x * len_y

    #gprMax
    def forward(self, path_input="/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt",
                path_gprMax="/home/yzi/research/gprMax",
                path_output="/home/yzi/research/GPR-FWI-2021/forward/forward_output_gt"):
        """run forward model based on the in txt files"""
        for i in range(self.num_shot):
            file = path_input + '/cross_well_cylinder_B_scan_shot_' + str(i) + '.in'# input txt file of gprMax
            os.system(#'eval "$(conda shell.bash hook)"\n'
                      # + 'conda activate gprMax\n'
                      'cd ' + path_gprMax + '\n'
                      + 'python -m gprMax ' + file + '\n'
                      + 'mv ' + file.split('.in')[0] + '.out ' + path_output)

    def make_in_gprMax_gt(self,path_input ="/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt" ):
        """generate txt files for forward based on the ground truth geo-model"""
        folder_path = path_input
        for i in range(self.num_shot):
            path_file = folder_path + '/cross_well_cylinder_B_scan_shot_' + str(i) + '.in'
            file = open(path_file, "w")
            # input current parameter=>forward=> current observation
            head_lines = ["#title: cross boreholes B-scan from two cylinders buried in ground\n",
                          "#domain: 8.0 8.0 0.1\n",
                          "#dx_dy_dz: 0.1 0.1 0.1\n",
                          "#time_window: 10e-08\n\n"
                          ]
            offset = 0.5
            shot_start = 1.0
            shot_index = i * offset + shot_start
            shot_receiver_line = ["#waveform: ricker 1.0 100000000.0 mysource\n",
                                  "#hertzian_dipole: z 1.0 " + str(shot_index) + " 0.0 mysource\n",
                                  "#rx_array: 7.0 1.0 0.0 7.0 7.0 0.0 0.0 0.5 0.0\n\n"]
            # build geo model
            material_lines = ["#material: 5.5 0.0028 1.0 0.0 mid\n",
                              "#material: 5.0 0.001 1.0 0.0 top\n",
                              "#material: 6.5 0.008 1.0 0.0 pipe\n",
                              "#material: 5.0 0.001 1.0 0.0 bg\n\n"]

            geo_model_lines = ["#box: 0.0 0.0 0.0 8.0 8.0 0.1 top\n",
                               "#cylindrical_sector: z 0.0 18.0 0.0 0.1 16.0 270.0 90.0 mid\n",
                               "#box: 0.0 6.0 0.0 8.0 8.0 0.1 top\n",
                               "#cylinder: 3.0 4.0 0.0 3.0 4.0 0.1 0.5 pipe\n",
                               "#cylinder: 5.0 4.0 0.0 5.0 4.0 0.1 0.5 pipe\n",
                               "#geometry_view: 0.0 0.0 0.0 8.0 8.0 0.1 0.1 0.1 0.1 cross_well_half_space n\n"]

            file.writelines(head_lines)
            file.writelines(shot_receiver_line)
            file.writelines(material_lines)
            file.writelines(geo_model_lines)
            file.close()

    def make_in_gprMax(self, path_output, file_para_p='./in_p.dat', file_para_c='./in_c.dat'):
        """generate txt files for forward based on the current estimated parameters of geo-model
        file_para_p/file_para_c is in.dat file which is PEST's output after inversion and input for gprMax for forward
        path_output point to the folder store these gprMax instruction txt files
        geo model is len_x*len_y matrix, each pixel has two parameters(permitivity(i,j) and conductivity(i,j))"""

        f1 = open(file_para_p, 'r')
        f1_Lines = f1.readlines()
        permittivity = np.array([float(i) for i in f1_Lines]).reshape(self.len_x, self.len_y)

        f2 = open(file_para_c, 'r')
        f2_Lines = f2.readlines()
        conductivity = np.array([float(i) for i in f2_Lines]).reshape(self.len_x, self.len_y)

        material_lines=[]
        geo_model_lines=[]
        for i in range(self.len_x):
            for j in range(self.len_y):
                x_l = i * 0.1
                x_r = (i + 1) * 0.1
                y_l = j * 0.1
                y_r = (j + 1) * 0.1
                x_l = ("%.2f" % x_l)
                x_r = ("%.2f" % x_r)
                y_l = ("%.2f" % y_l)
                y_r = ("%.2f" % y_r)
                index_pix = "pix_"+str(i)+"_"+str(j)
                material_lines.append("#material: "+str(permittivity[i][j])+" "+str(conductivity[i][j])+" 1.0 0.0 "+index_pix+"\n")
                geo_model_lines.append("#box: "+x_l+" "+y_l+" 0.0 "+x_r+" "+y_r+" 0.1 "+index_pix+"\n")
        material_lines[-1] = material_lines[-1]+"\n"

        folder_path = path_output
        for _ in range(self.num_shot):
            path_file = folder_path + '/cross_well_cylinder_B_scan_shot_' + str(_) + '.in'
            file = open(path_file, "w")
            # input current parameter=>forward=> current observation
            head_lines = ["#title: cross boreholes B-scan from two cylinders buried in ground\n",
                          "#domain: 8.0 8.0 0.1\n",
                          "#dx_dy_dz: 0.1 0.1 0.1\n",
                          "#time_window: 10e-08\n\n"
                          ]
            offset = 0.5
            shot_start = 1.0
            shot_index = _ * offset + shot_start
            shot_receiver_line = ["#waveform: ricker 1.0 100000000.0 mysource\n",
                                  "#hertzian_dipole: z 1.0 " + str(shot_index) + " 0.0 mysource\n",
                                  "#rx_array: 7.0 1.0 0.0 7.0 7.0 0.0 0.0 0.5 0.0\n\n"]

            file.writelines(head_lines)
            file.writelines(shot_receiver_line)
            file.writelines(material_lines)
            file.writelines(geo_model_lines)
            file.close()


    # PEST
    def inverse(self,path_folder_pest, file_pst):
        """run inversion given bash of forward model"""
        os.system('export PATH=$PATH:' + path_folder_pest+'\n'+'pest '+file_pst+'\n')

    def make_in_tpl_pest(self, tpl_file, mode='permittivity'):
        """make the tpl file for permittivity/conductivity inversion"""
        f = open(tpl_file, "w")
        in_tpl_headlines = ["ptf #\n"]
        if mode == 'permittivity':
            para = "p"
        elif mode == 'conductivity':
            para = "c"
        else:
            sys.exit("input of function is wrong!")
        for i in range(self.len_x):
            for j in range(self.len_y):
                name_para = para + str(i) + "_" + str(j)
                in_tpl_headlines.append("# " + name_para + "        #\n")
        f.writelines(in_tpl_headlines)
        f.close()

    def make_in_pest(self, in_data_file, inital_value=5.5):
        """in.data store the value of parameter to estimate"""
        f = open(in_data_file, "w")
        in_data_headlines = []
        inital_value = ("%e" % inital_value)
        for i in range(self.len_x):
            for j in range(self.len_y):
                in_data_headlines.append(str(inital_value) + "\n")
        f.writelines(in_data_headlines)
        f.close()


    def update_curr_obs(self, path_folder_h5_result):
        """read gprMax's hd5f result to the matrix self.curr_obs [shot,receiver,len_sig]"""
        path = path_folder_h5_result + '/cross_well_cylinder_B_scan_shot_'
        for i in range(self.num_shot):
            path_f = path + str(i) + '.out'
            h5_file = h5py.File(path_f, 'r')
            rxs = h5_file['rxs']
            for j in range(self.num_receiver):
                rx = rxs['rx' + str(j + 1)]
                self.curr_obs[i][j][:] = rx['Ez'][()]

    def update_gt_obs(self, path_folder_h5_result):
        """read gprMax's hd5f result to the matrix self.curr_obs [shot,receiver,len_sig]"""
        path = path_folder_h5_result + '/cross_well_cylinder_B_scan_shot_'
        for i in range(self.num_shot):
            path_f = path + str(i) + '.out'
            h5_file = h5py.File(path_f, 'r')
            rxs = h5_file['rxs']
            for j in range(self.num_receiver):
                rx = rxs['rx' + str(j + 1)]
                self.real_obs[i][j][:] = rx['Ez'][()]

    def make_curr_obs_pest(self, file_out_data):
        """out.data"""
        f = open(file_out_data, "w")
        # input current parameter=>forward=> current observation
        cur_obs_data_lines = [""]
        for i in range(self.num_shot):
            for j in range(self.num_receiver):
                for k in range(self.length_sig):
                    formator = ("%e" % self.curr_obs[i][j][k])
                    cur_obs_data_lines.append(str(formator) + "\n")
        f.writelines(cur_obs_data_lines)
        f.close()  # to change file access modes

    def make_ins_pest(self, ins_file):
        """out.ins"""
        f = open(ins_file, "w")
        # input current parameter=>forward=> current observation
        out_ins_lines = ["pif #\n"]
        for i in range(self.num_shot):
            for j in range(self.num_receiver):
                for k in range(self.length_sig):
                    name_obs = "o" + str(i) + "_" + str(j) + "_" + str(k)
                    # out_ins_lines.append("l1 (" + name_obs + ")1:12\n")
                    out_ins_lines.append("l1 !" + name_obs + "!\n")
        f.writelines(out_ins_lines)
        f.close()  # to change file access modes

    def make_pst_pest(self, pst_file, mode='permittivity', initial_para_value=np.zeros((80,80)) + 5.5):
        """make the .pst control file for PEST permittivity/conductivity inversion

        input:
        initial value of estimated parameters;
        real observation[shot, receriver, len_sig];
        .pst file

        output:
        .pst file"""
        f = open(pst_file, "w")
        if mode == 'permittivity':
            para = "p"
        elif mode == 'conductivity':
            para = "c"
        else:
            sys.exit("input of function is wrong!")
        ctrl_data_lines = ["pcf\n", "* control data\n", "restart  estimation\n",
                           "    6400    71825     6400     0     1\n",
                           "    1     1 single point   1   0   0\n", "  5.0   2.0   0.3  0.03    10\n",
                           "  3.0   3.0 0.001\n", "  0.1\n   3  0.01     3     3  0.01     3\n", "    0     0     0\n"]
        # parameter groups
        # initial parameter data
        para_lines = ["* parameter groups\n"]
        para_data_lines = ["* parameter data\n"]
        for i in range(self.len_x):
            for j in range(self.len_y):
                name_para = para + str(i) + "_" + str(j)
                para_lines.append(name_para + "           relative 0.01  0.0  switch  2.0 parabolic\n")
                para_data_lines.append(
                    name_para + "           none relative   "+str(initial_para_value[i][j])+"00000      -1.000000E+10   1.000000E+10 "
                    + name_para + "              1.0000        0.0000      1\n")
        obs_groups_lines = ["* observation groups\nobsgroup\n"]
        obs_data_lines = ["* observation data\n"]
        for i in range(self.num_shot):
            for j in range(self.num_receiver):
                for k in range(self.length_sig):
                    name_obs_para = "o" + str(i) + "_" + str(j) + "_" + str(k)
                    obs_f = ("%.6f" % self.real_obs[i][j][k])
                    obs_data_lines.append(name_obs_para + "            " + str(obs_f) + "      1.0  obsgroup\n")

        model_control_lines = ["* model command line\n", "bash "+para+"_forward.sh\n", "* model input/output\n",
                               "in"+"_"+para+".tpl in"+"_"+para+".dat\n", "out"+"_"+para+".ins out"+"_"+para+".dat\n", "* prior information\n"]
        f.writelines(ctrl_data_lines)
        f.writelines(para_lines)
        f.writelines(para_data_lines)
        f.writelines(obs_groups_lines)
        f.writelines(obs_data_lines)
        f.writelines(model_control_lines)
        f.close()  # to change file access modes
