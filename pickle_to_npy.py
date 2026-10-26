import pickle
import numpy as np

LOADPATH = "/work/kajiyama/preprocessed/APHRODITE" \
           "/aphrodite_anom_std_interp.pickle"
SAVEDIR = "/work/kajiyama/cnn/transfer_input/pr/main"

# keys written by preprocess.py, saved as aphro_<suffix>.npy
KEY_LST = ["pr_raw",
           "pr_clim",
           "pr_variance",
           "pr_anom",
           "pr_std",
           "pr_coarse",
           "pr_coarse_clim",
           "pr_coarse_variance",
           "pr_coarse_anom",
           "pr_coarse_std"]

def main():
    save_flag = True
    data = load(LOADPATH)
    mk_newarr(data, save_flag=save_flag)

def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def mk_newarr(data, save_flag=False):
    if save_flag is True:
        for key in KEY_LST:
            name = "aphro_" + key[len("pr_"):]
            savepath = f"{SAVEDIR}/{name}.npy"
            np.save(savepath, data[key])
            print(f"{savepath} is saved")


if __name__ == "__main__":
    main()
