
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

from utiles import tools
import numpy as np
import os

def main():
    # forward ground truth geo model to get observation of ground truth
    agent = tools(num_shot=5, num_receiver=5, length_sig=425, len_x=50, len_y=50, scale_discount=5)

    # output gprMax.in files
    agent.make_in_gprMax_gt(path_input ="./output/forward/forward_input_gt")
    # output gprMax.out files
    agent.forward(path_input="/homelocal/GPR-FWI-2021/output/forward/forward_input_gt",
                path_gprMax="/homelocal/gprMax",
                path_output="/homelocal/GPR-FWI-2021/output/forward/forward_output_gt")

    # read gprMax.out files as the gt observation matrix
    agent.update_gt_obs(path_folder_h5_result="./output/forward/forward_output_gt")

    # permittivity inversion
    # make inversion control file based on the gt observation and initial model
    agent.make_pst_pest(pst_file="./output/inversion/inversion_p.pst",
                        mode='permittivity',
                        initial_value=5.5)
    # make the instruction file for current observation for inversion
    agent.make_ins_pest(ins_file="./output/inversion/out_p.ins")
    # make the input template file for current model's parameters for inversion
    agent.make_in_tpl_pest(tpl_file="./output/inversion/in_p.tpl", mode='permittivity')
    # make the input file for initial model's parameters for inversion
    agent.make_in_pest(in_data_file="./output/inversion/in_p.dat", inital_value=5.5)

    # update current observation after forward/permittivity then inverse conductivity
    # conductivity inverse
    # make inversion control file based on the gt observation and initial model
    # agent.make_pst_pest(pst_file="./output/inversion/inversion_c.pst",
    #                     mode='conductivity',
    #                     initial_value=0.0028)
    # # make the instruction file for current observation for inversion
    # agent.make_ins_pest(ins_file="./output/inversion/out_c.ins")
    # # make the input template file for current model's parameters for inversion
    # agent.make_in_tpl_pest(tpl_file="./output/inversion/in_c.tpl", mode='conductivity')
    # agent.make_in_pest(in_data_file="./output/inversion/in_c.dat", inital_value=0.0028)


    # make the input file for initial model's parameters for inversion
    # forward with these initial parameters runed by the pest with forward bash

    agent.inverse(path_folder_pest="/homelocal/pest_source",
                  path_inversion="/homelocal/GPR-FWI-2021/output/inversion/",
                  file_pst="inversion_p.pst")
    # agent.inverse(path_folder_pest="/homelocal/pest_source",
    #               path_inversion="/homelocal/GPR-FWI-2021/output/inversion/",
    #               file_pst="inversion_c.pst")
    epoch = 0
    ex_log_folder = './output/experiments/ex_0'
    os.system('cd ' + ex_log_folder + '\n' + 'mkdir epoch_' + str(epoch) + '\n')
    agent.plot_section(para_file='./output/inversion/in_p.dat',
                     ex_log_folder='./output/experiments/ex_0',
                     num_epoch=0)
    # agent.plot_section(para_file='./output/inversion/in_c.dat',
    #                  ex_log_folder='./output/experiments/ex_0',
    #                  num_epoch=0)


    # use first inversion c and p as the new initial model
    for epoch in range(3):
        agent.update_gt_obs(path_folder_h5_result="./output/forward/forward_output_gt")
        agent.update_pst_pest(pst_file="./output/inversion/inversion_p.pst",
                            mode='permittivity',
                            start_para_file='/homelocal/GPR-FWI-2021/output/inversion/in_p.dat')
        agent.inverse(path_folder_pest="/homelocal/pest_source",
                      path_inversion="/homelocal/GPR-FWI-2021/output/inversion/",
                      file_pst="inversion_p.pst")


        agent.update_pst_pest(pst_file="./output/inversion/inversion_c.pst",
                            mode='conductivity',
                            start_para_file='/homelocal/GPR-FWI-2021/output/inversion/in_c.dat')

        agent.inverse(path_folder_pest="/homelocal/pest_source",
                      path_inversion="/homelocal/GPR-FWI-2021/output/inversion/",
                      file_pst="inversion_c.pst")

        ex_log_folder = './output/experiments/ex_0'
        os.system('cd ' + ex_log_folder + '\n' + 'mkdir epoch_' + str(epoch + 1) + '\n')
        agent.plot_section(para_file='./output/inversion/in_p.dat',
                           ex_log_folder='./output/experiments/ex_0',
                           num_epoch=epoch + 1)
        # agent.plot_section(para_file='./output/inversion/in_c.dat',
        #                    ex_log_folder='./output/experiments/ex_0',
        #                    num_epoch=epoch + 1)



if __name__ == "__main__":
    main()
# # %%
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import fftpack
# time_range = 10e-08
# time_step = time_range/425
# fs = 1/time_step
# time_vec =  np.arange(0, time_range, time_step)
# f = h5py.File('./output/forward/forward_output_gt/cross_well_cylinder_B_scan_shot_3.out')
# rxs = f['rxs']
# rx1 = rxs['rx1']
# signal = rx1['Ez']
# signal = signal.value
# plt.plot(time_vec, signal)
# plt.show()
# plt.clf()
# plt.close()
# plt.cla
# # %%
# sig_fft = fftpack.fft(signal)
# sample_freq = fftpack.fftfreq(sig_fft.size, d=time_step)
# plt.plot(sample_freq[:30], abs(sig_fft)[:30])
# plt.show()
# plt.clf()
# plt.close()
# plt.cla



