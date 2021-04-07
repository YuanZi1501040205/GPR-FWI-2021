
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

from utiles import tools

def main():
    agent = tools(num_shot=5, num_receiver=5, length_sig=425, len_x=30, len_y=30, scale_discount=3)
    agent.make_in_gprMax(path_output="/homelocal/GPR-FWI-2021/output/forward/forward_input",
                         file_para_p='/homelocal/GPR-FWI-2021/output/inversion/in_p.dat',
                         file_para_c='/homelocal/GPR-FWI-2021/output/inversion/in_c.dat')

    agent.forward(path_input="/homelocal/GPR-FWI-2021/output/forward/forward_input",
                path_gprMax="/homelocal/gprMax",
                path_output="/homelocal/GPR-FWI-2021/output/forward/forward_output")

    agent.update_curr_obs(path_folder_h5_result = "/homelocal/GPR-FWI-2021/output/forward/forward_output")

    agent.make_curr_obs_pest(file_out_data='/homelocal/GPR-FWI-2021/output/inversion/out_c.dat')


if __name__ == "__main__":
    main()
