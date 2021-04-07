import numpy as np
import os
import sys
import h5py

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
                      + 'cd ' + path_gprMax + '\n'
                      + 'python -m gprMax ' + file + '\n'
                      + 'mv ' + file.split('.in')[0] + '.out ' + path_output)



    def make_in_gprMax(self, path_output, file_para_p='./in_p.dat', file_para_c='./in_c.dat', ):
        """generate txt files for forward based on the current estimated parameters of geo-model
        file_para_p/file_para_c is in.dat file which is PEST's output after inversion and input for gprMax for forward
        path_output point to the folder store these gprMax instruction txt files
        geo model is len_x*len_y matrix, each pixel has two parameters(permitivity(i,j) and conductivity(i,j))"""
        x_scale = int(self.len_x/self.scale_discount)
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
                          "#domain: 3.0 3.0 0.1\n",
                          "#dx_dy_dz: 0.1 0.1 0.1\n",
                          "#time_window: 1e-07\n\n"
                          ]
            offset = 0.5
            shot_start = 0.5
            shot_index = _ * offset + shot_start
            shot_receiver_line = ["#waveform: ricker 1.0 5e7 mysource\n",
                                  "#hertzian_dipole: z 0.5 " + str(shot_index) + " 0.0 mysource\n",
                                  "#rx_array: 2.5 0.5 0.0 2.5 2.5 0.0 0.0 0.5 0.0\n\n"]

            file.writelines(head_lines)
            file.writelines(shot_receiver_line)
            file.writelines(material_lines)
            file.writelines(geo_model_lines)
            file.close()




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

