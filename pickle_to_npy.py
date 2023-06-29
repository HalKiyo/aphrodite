import pickle
import numpy as np
import numpy.ma as ma

def main():
    save_flag = True
    savepath = "/work/kajiyama/preprocessed/APHRODITE" \
                "/aphrodite_anom_std_interp.pickle"
    data = load(savepath)
    mk_newarr(data, save_flag=save_flag)

def load(file):
    with open(file, 'rb') as f:
        data =pickle.load(f)
    return data

def mk_newarr(data, save_flag=False):
    savedir = f"/work/kajiyama/cnn/transfer_input/pr/main"
    key_lst = ["pr_raw", 
               "pr_clim",
               "pr_variance",
               "pr_anom",
               "pr_std",
               "pr_coarse",
               "pr_coarse_clim",
               "pr_coarse_variance",
               "pr_coarse_anom",
               "pr_coarse_std"]

    nam_lst = ["aphro_raw",
               "aphro_clim",
               "aphro_variance",
               "aphro_anom",
               "aphro_std",
               "aphro_coarse",
               "aphro_coarse_clim",
               "aphro_coarse_variance",
               "aphro_coarse_anom",
               "aphro_coarse_std"]

    if save_flag is True:
        for name, key in zip(nam_lst, key_lst):
            savepath = f"{savedir}/{name}.npy"
            var = data[key]
            np.save(savepath, var)
            print(f"{savepath} is saved")


if __name__ == "__main__":
    main()
