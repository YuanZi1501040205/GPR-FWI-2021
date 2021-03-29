
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

from utiles import tools
import numpy as np

def main():
    # forward ground truth geo model to get observation of ground truth
    agent = tools(num_shot=13, num_receiver=13, length_sig=425, len_x=80, len_y=80, scale_discount=5)
    # agent.make_in_gprMax_gt(path_input ="./output/forward/forward_input_gt")
    # agent.forward(path_input="/homelocal/GPR-FWI-2021/output/forward/forward_input_gt",
    #             path_gprMax="/homelocal/gprMax",
    #             path_output="/homelocal/GPR-FWI-2021/output/forward/forward_output_gt")
    # agent.update_gt_obs(path_folder_h5_result="./output/forward/forward_output_gt")
    # # make control files
    # # permittivity
    # agent.make_pst_pest(pst_file="./output/inversion/inversion_p.pst",
    #                     mode='permittivity',
    #                     initial_value=5.5)
    #
    # # permittivity
    # agent.make_pst_pest(pst_file="./output/inversion/inversion_c.pst",
    #                     mode='conductivity',
    #                     initial_value=0.0028)
    #
    # # initial permittivity and initial conductivity
    # # permittivity
    # agent.make_ins_pest(ins_file="./output/inversion/out_p.ins")
    # agent.make_in_tpl_pest(tpl_file="./output/inversion/in_p.tpl", mode='permittivity')
    # agent.make_in_pest(in_data_file="./output/inversion/in_p.dat", inital_value=5.5)
    # # conductivity
    # agent.make_ins_pest(ins_file="./output/inversion/out_c.ins")
    # agent.make_in_tpl_pest(tpl_file="./output/inversion/in_c.tpl", mode='conductivity')
    # agent.make_in_pest(in_data_file="./output/inversion/in_c.dat", inital_value=0.0028)

    # forward with these initial parameters runed by the pest with forward bash
    agent.inverse(path_folder_pest="/homelocal/pest_source",
                  path_inversion="/homelocal/GPR-FWI-2021/output/inversion/",
                  file_pst="inversion_p.pst")

    # # update current observation after forward/permittivity
    # agent.update_curr_obs(path_folder_h5_result="./output/forward/forward_output")
    # agent.make_curr_obs_pest(file_out_data="./output/inversion/out_p.dat")
    # # update current observation after forward/conductivity
    # agent.update_curr_obs(path_folder_h5_result="./output/forward/forward_output")
    # agent.make_curr_obs_pest(file_out_data="./output/inversion/out_c.dat")




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


