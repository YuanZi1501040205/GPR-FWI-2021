
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

from utiles import tools

def main():
    agent = tools(num_shot=13, num_receiver=13, length_sig=425, len_x=80, len_y=80)
    agent.make_in_gprMax(path_output="/home/yzi/research/GPR-FWI-2021/output/forward/forward_input",
                         file_para_p='/home/yzi/research/GPR-FWI-2021/output/inversion/in_p.dat',
                         file_para_c='/home/yzi/research/GPR-FWI-2021/output/inversion/in_c.dat')

    agent.forward(path_input="/home/yzi/research/GPR-FWI-2021/output/forward/forward_input",
                path_gprMax="/home/yzi/research/gprMax",
                path_output="/home/yzi/research/GPR-FWI-2021/output/forward/forward_output")

    agent.update_curr_obs(path_folder_h5_result = "/home/yzi/research/GPR-FWI-2021/output/forward/forward_output")

    agent.make_curr_obs_pest(file_out_data='/home/yzi/research/GPR-FWI-2021/output/inversion/out_c.dat')


if __name__ == "__main__":
    main()
