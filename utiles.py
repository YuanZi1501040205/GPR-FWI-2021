import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import cv2

class tools():
    def __init__(self, num_shot=13, num_receiver=13, length_sig=425, len_x=80, len_y=80, scale_discount=5):
        self.num_shot = num_shot
        self.num_receiver = num_receiver
        self.length_sig = length_sig
        self.len_x = len_x
        self.len_y = len_y
        self.real_obs = np.zeros((num_shot, num_receiver, length_sig))
        self.curr_obs = np.zeros((num_shot, num_receiver, length_sig))
        self.num_obs = num_shot * num_receiver * length_sig
        self.num_para = len_x * len_y
        self.scale_discount = scale_discount

    #gprMax
    def forward(self, path_input="/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt",
                path_gprMax="/home/yzi/research/gprMax",
                path_output="/home/yzi/research/GPR-FWI-2021/forward/forward_output_gt"):
        """run forward model based on the in txt files"""
        for i in range(self.num_shot):
            file = path_input + '/cross_well_cylinder_B_scan_shot_' + str(i) + '.in'# input txt file of gprMax
            os.system(#'eval "$(conda shell.bash hook)"\n'
                      # + 'conda deactivate\n'
                      # + 'conda activate gprMax\n'
                        'export PATH=$PATH:/usr/local/cuda-11.1/bin\n'
                        +'export PATH=$PATH:/usr/local/cuda/bin/\n'
                      + 'cd ' + path_gprMax + '\n'
                      + 'python -m gprMax ' + file + ' -gpu 2\n'
                      + 'mv ' + file.split('.in')[0] + '.out ' + path_output)

    # def make_in_gprMax_gt(self,path_input ="/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt" ):
    #     """generate txt files for forward based on the ground truth geo-model"""
    #     folder_path = path_input
    #     for i in range(self.num_shot):
    #         path_file = folder_path + '/cross_well_cylinder_B_scan_shot_' + str(i) + '.in'
    #         file = open(path_file, "w")
    #         # input current parameter=>forward=> current observation
    #         head_lines = ["#title: cross boreholes B-scan from two cylinders buried in ground\n",
    #                       "#domain: 8.0 8.0 0.1\n",
    #                       "#dx_dy_dz: 0.1 0.1 0.1\n",
    #                       "#time_window: 3e-07\n\n"
    #                       ]
    #         offset = 0.5
    #         shot_start = 1.0
    #         shot_index = i * offset + shot_start
    #         shot_receiver_line = ["#waveform: ricker 1.0 10000000.0 mysource\n",
    #                               "#hertzian_dipole: z 1.0 " + str(shot_index) + " 0.0 mysource\n",
    #                               "#rx_array: 7.0 1.0 0.0 7.0 7.0 0.0 0.0 0.5 0.0\n\n"]
    #
    #         # build geo model
    #         material_lines = ["#material: 5.5 0.0028 1.0 0.0 mid\n",
    #                           "#material: 5.0 0.001 1.0 0.0 top\n",
    #                           "#material: 6.5 0.008 1.0 0.0 pipe\n",
    #                           "#material: 5.0 0.001 1.0 0.0 bg\n\n"]
    #
    #         geo_model_lines = ["#box: 0.0 0.0 0.0 8.0 8.0 0.1 top\n",
    #                            "#cylindrical_sector: z 0.0 18.0 0.0 0.1 16.0 270.0 90.0 mid\n",
    #                            "#box: 0.0 6.0 0.0 8.0 8.0 0.1 top\n",
    #                            "#cylinder: 3.0 4.0 0.0 3.0 4.0 0.1 1 pipe\n",
    #                            "#cylinder: 5.0 4.0 0.0 5.0 4.0 0.1 1 pipe\n",
    #                            "#geometry_view: 0.0 0.0 0.0 8.0 8.0 0.1 0.1 0.1 0.1 cross_well_half_space n\n"]
    #
    #         file.writelines(head_lines)
    #         file.writelines(shot_receiver_line)
    #         file.writelines(material_lines)
    #         file.writelines(geo_model_lines)
    #         file.close()

    #

    def make_in_gprMax_gt(self,path_input = "/home/yzi/research/GPR-FWI-2021/forward/forward_input_gt" ):
        """generate txt files for forward based on the ground truth geo-model"""
        folder_path = path_input
        for i in range(self.num_shot):
            path_file = folder_path + '/cross_well_cylinder_B_scan_shot_' + str(i) + '.in'
            file = open(path_file, "w")
            # input current parameter=>forward=> current observation
            head_lines = ["#title: cross boreholes B-scan from two cylinders buried in ground\n",
                          "#domain: 5.0 5.0 0.1\n",
                          "#dx_dy_dz: 0.1 0.1 0.1\n",
                          "#time_window: 1e-07\n\n"
                          ]
            offset = 0.5
            shot_start = 1.5
            shot_index = i * offset + shot_start
            shot_index =  ("%.1f"% shot_index)
            shot_receiver_line = ["#waveform: ricker 1.0 5e7 mysource\n",
                                  "#hertzian_dipole: z 1.5 " + str(shot_index) + " 0.0 mysource\n",
                                  "#rx_array: 3.5 1.5 0.0 3.5 3.5 0.0 0.0 0.5 0.0\n\n"]

            # build geo model
            material_lines = ["#material: 5.5 0.0028 1.0 0.0 mid\n",
                              # "#material: 5.0 0.001 1.0 0.0 top\n",
                              # "#material: 6.5 0.008 1.0 0.0 pipe\n",
                              "#material: 6.5 0.0028 1.0 0.0 pipe\n"
                             ]

            geo_model_lines = [
                               "#box: 1.5 0.0 0.0 3.5 5.0 0.1 mid\n",
                                "#box: 3.0 3.0 0.0 4.0 4 0.1 pipe\n",
                               # "#cylinder: 2.5 2.5 0.0 2.5 2.5 0.1 0.5 pipe\n",
                               "#geometry_view: 0.0 0.0 0.0 5.0 5.0 0.1 0.1 0.1 0.1 cross_well_half_space n\n"]

            file.writelines(head_lines)
            file.writelines(shot_receiver_line)
            file.writelines(material_lines)
            file.writelines(geo_model_lines)
            file.close()

    def make_in_gprMax(self, path_output, file_para_p='./in_p.dat', file_para_c='./in_c.dat', ):
        """generate txt files for forward based on the current estimated parameters of geo-model
        file_para_p/file_para_c is in.dat file which is PEST's output after inversion and input for gprMax for forward
        path_output point to the folder store these gprMax instruction txt files
        geo model is len_x*len_y matrix, each pixel has two parameters(permitivity(i,j) and conductivity(i,j))"""
        x_scale = int(self.len_x/ self.scale_discount)
        y_scale = int(self.len_y / self.scale_discount)
        f1 = open(file_para_p, 'r')
        f1_Lines = f1.readlines()
        permittivity = np.array([float(i) for i in f1_Lines]).reshape(x_scale, y_scale)

        f2 = open(file_para_c, 'r')
        f2_Lines = f2.readlines()
        conductivity = np.array([float(i) for i in f2_Lines]).reshape(x_scale, y_scale)

        material_lines=[]
        geo_model_lines=[]

        for i in range(x_scale):
            for j in range(y_scale):
                if i < 3 or i >= 7 or j < 3 or j >= 7:
                    material_lines.append("\n")
                    geo_model_lines.append("\n")
                else:
                    x_l = i * 0.1 * self.scale_discount
                    x_r = (i + 1) * 0.1 * self.scale_discount
                    y_l = j * 0.1 * self.scale_discount
                    y_r = (j + 1) * 0.1 * self.scale_discount
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
                          "#domain: 5.0 5.0 0.1\n",
                          "#dx_dy_dz: 0.1 0.1 0.1\n",
                          "#time_window: 1e-07\n\n"
                          ]
            offset = 0.5
            shot_start = 1.5
            shot_index = _ * offset + shot_start
            shot_index =  ("%.1f"% shot_index)
            shot_receiver_line = ["#waveform: ricker 1.0 5e7 mysource\n",
                                  "#hertzian_dipole: z 1.5 " + str(shot_index) + " 0.0 mysource\n",
                                  "#rx_array: 3.5 1.5 0.0 3.5 3.5 0.0 0.0 0.5 0.0\n\n"]

            file.writelines(head_lines)
            file.writelines(shot_receiver_line)
            file.writelines(material_lines)
            file.writelines(geo_model_lines)
            file.close()


    # PEST
    def inverse(self,path_folder_pest, path_inversion, file_pst):
        """run inversion given bash of forward model"""
        os.system('cd '+path_inversion+'\n'+'export PATH=$PATH:' + path_folder_pest+'\n'+'pest '+file_pst+'\n')

    def make_in_tpl_pest(self, tpl_file, mode='permittivity'):
        """make the tpl file for permittivity/conductivity inversion"""
        x_scale = int(self.len_x / self.scale_discount)
        y_scale = int(self.len_y / self.scale_discount)
        f = open(tpl_file, "w")
        in_tpl_headlines = ["ptf #\n"]
        if mode == 'permittivity':
            para = "p"
        elif mode == 'conductivity':
            para = "c"
        else:
            sys.exit("input of function is wrong!")
        for i in range(x_scale):
            for j in range(y_scale):
                name_para = para + str(i) + "_" + str(j)
                in_tpl_headlines.append("# " + name_para + "        #\n")
        f.writelines(in_tpl_headlines)
        f.close()

    def make_in_pest(self, in_data_file, inital_value=5.5):
        """in.data store the value of parameter to estimate"""
        x_scale = int(self.len_x/self.scale_discount)
        y_scale = int(self.len_y / self.scale_discount)
        f = open(in_data_file, "w")
        in_data_headlines = []
        inital_value = ("%e" % inital_value)
        for i in range(x_scale):
            for j in range(y_scale):
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

    def make_pst_pest(self, pst_file, mode='permittivity', initial_value=5.5):
        """make the .pst control file for PEST permittivity/conductivity inversion

        input:
        initial value of estimated parameters;
        real observation[shot, receriver, len_sig];
        .pst file

        output:
        .pst file"""
        x_scale = int(self.len_x/self.scale_discount)
        y_scale = int(self.len_y / self.scale_discount)
        initial_para_value = np.zeros((x_scale, y_scale)) + initial_value
        f = open(pst_file, "w")
        if mode == 'permittivity':
            para = "p"
            update_interval = 0.1 # from 5.5->6.5
            low_bound = 1
            high_bound = 10
        elif mode == 'conductivity':
            para = "c"
            update_interval = 0.0001
            low_bound = 0.0001
            high_bound = 0.01
        else:
            sys.exit("input of function is wrong!")
        ctrl_data_lines = ["pcf\n", "* control data\n", "restart  estimation\n",
                           "    "+str(int(x_scale * y_scale))+"    "+str(self.num_shot * self.num_shot *self.length_sig)
                           +"     "+str(int(x_scale * y_scale))+"     0     1\n",
                           "    1     1 single point   1   0   0\n", "  5.0   2.0   0.3  0.03    10\n",
                           "  3.0   3.0 0.001\n", "  0.1\n   20  0.01     3     3  0.01     3\n", "    0     0     0\n"]
        # parameter groups
        # initial parameter data
        para_lines = ["* parameter groups\n"]
        para_data_lines = ["* parameter data\n"]
        for i in range(x_scale):
            for j in range(y_scale):
                name_para = para + str(i) + "_" + str(j)
                para_lines.append(name_para + "           relative "+str(update_interval)+"  0.0  switch  2.0 parabolic\n")
                value = ("%.5f" % initial_para_value[i][j])
                para_data_lines.append(
                    name_para + "           none relative   "+str(value)+"00000      "+str(low_bound)+"   "+str(high_bound)+" "
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

    def update_pst_pest(self, pst_file, mode='permittivity', start_para_file='/homelocal/GPR-FWI-2021/output/inversion/in_p.dat'):
        """make the .pst control file for PEST permittivity/conductivity inversion

        input:
        initial value of estimated parameters;
        real observation[shot, receriver, len_sig];
        .pst file

        output:
        .pst file"""
        x_scale = int(self.len_x/self.scale_discount)
        y_scale = int(self.len_y / self.scale_discount)
        f_initial = open(start_para_file, "r")
        f_initial_Lines = f_initial.readlines()
        initial_value = np.zeros((x_scale, y_scale))
        for i in range(x_scale):
            for j in range(y_scale):
                initial_value[i][j] = float(f_initial_Lines[i * x_scale + j])

        initial_para_value = initial_value
        f = open(pst_file, "w")
        if mode == 'permittivity':
            para = "p"
            update_interval = 0.1 # from 5.5->6.5
            low_bound = 1
            high_bound = 10
        elif mode == 'conductivity':
            para = "c"
            update_interval = 0.0001
            low_bound = 0.0001
            high_bound = 0.01
        else:
            sys.exit("input of function is wrong!")
        ctrl_data_lines = ["pcf\n", "* control data\n", "restart  estimation\n",
                           "    "+str(int(x_scale * y_scale))+"    "+str(self.num_shot * self.num_shot *self.length_sig)
                           +"     "+str(int(x_scale * y_scale))+"     0     1\n",
                           "    1     1 single point   1   0   0\n", "  5.0   2.0   0.3  0.03    10\n",
                           "  3.0   3.0 0.001\n", "  0.1\n   20  0.01     3     3  0.01     3\n", "    0     0     0\n"]
        # parameter groups
        # initial parameter data
        para_lines = ["* parameter groups\n"]
        para_data_lines = ["* parameter data\n"]
        for i in range(x_scale):
            for j in range(y_scale):
                name_para = para + str(i) + "_" + str(j)
                para_lines.append(name_para + "           relative "+str(update_interval)+"  0.0  switch  2.0 parabolic\n")
                value = ("%.5f" % initial_para_value[i][j])
                para_data_lines.append(
                    name_para + "           none relative   " + str(value) + "00000      " + str(
                        low_bound) + "   " + str(high_bound) + " "
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

    def plot_section(self, para_file='./output/inversion/in_c.dat',
                     ex_log_folder='./output/figures/ex_2',
                     num_epoch=0):

        name_file = para_file.split('/')[-1].split('.dat')[0]
        x_scale = int(self.len_x / self.scale_discount)
        y_scale = int(self.len_y / self.scale_discount)
        file_para = para_file
        f1 = open(file_para, 'r')
        f1_Lines = f1.readlines()
        para_matrix = np.array([float(i) for i in f1_Lines]).reshape(x_scale, y_scale)

        path_figure = ex_log_folder + '/epoch_' + str(num_epoch) + '/' + name_file

        # Plot the shot gather
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        max_v = np.percentile(np.abs(para_matrix), 99)
        min_v = np.percentile(np.abs(para_matrix), 1)
        norm = colors.Normalize(vmin=-min_v, vmax=max_v, clip=True)
        para_rotate_90_clockwise = cv2.rotate(para_matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im = axes.imshow(para_rotate_90_clockwise, aspect='auto', norm=norm, cmap='seismic')
        axes.set_title(name_file + '_epoch_' + str(num_epoch))
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        fig.colorbar(im)
        title = 'FWI'
        fig.suptitle(title, verticalalignment='center')
        plt.savefig(os.path.join(path_figure + '.png'))
        plt.close()
        plt.cla()

        os.system(
            'cp ' + para_file + ' ' + ex_log_folder + '/epoch_' + str(num_epoch) + '\n'
            + 'cp ' + para_file.split('in_')[0] + 'inversion_' + para_file.split('in_')[-1].split('.')[
                0] + '.rec' + ' ' +
            ex_log_folder + '/epoch_' + str(num_epoch) + '\n')