import bisect
import numpy as np
import numpy.ma as ma
from os.path import exists
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def main(class_num=5, discrete_mode='EFD', resolution='1x1'):
    save_flag = False
    class_num = class_num
    discrete_mode = discrete_mode
    resolution = resolution
    if resolution == '1x1':
        lat_grid, lon_grid = 20, 20
        key = '1x1_coarse'

    workdir = f"/work/kajiyama/cnn/transfer_input/pr"
    fname = f"aphro_{key}_std_MJJASO_thailand"
    thailand_path = workdir + f"/continuous/thailand/{resolution}/{fname}.npy"
    thailand_spath = workdir + f"/class/thailand/{discrete_mode}" \
                     f"/{fname}_{discrete_mode}_{class_num}.npy"

    thailand = np.load(thailand_path) # thailand.shape=(65, 20, 20)

    if discrete_mode == 'EFD':
        thailand_class, thailand_bnd = thailand_EFD(thailand)
        print(f"thailand_bnd: {thailand_bnd}")
        save_npy(thailand_spath, thailand_class, save_flag=save_flag)

def _fill(x, mask_value=-99):
    f = ma.filled(x, fill_value=mask_value)
    return f

def _mask(x, mask_value=-99):
    m = ma.masked_where(x<=mask_value, x)
    return m

def save_npy(path, data, save_flag=False):
    if save_flag is True:
        np.save(path, data)
        print(f"class_outut has been SAVED")
    else:
        print(f"class_output is ***NOT*** saved yet")

def thailand_EFD(data, class_num=5, lat_grid=20, lon_grid=20):
    # EFD_bnd
    mjjaso_thailand = data.copy() # data=(65, 20, 20)
    thailand_flat = mjjaso_thailand.reshape(-1)
    flat_sorted = np.sort(thailand_flat)
    flat_sorted_masked = _mask(flat_sorted)
    count = int(len(flat_sorted) - ma.count_masked(flat_sorted_masked))
    if count%class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(count/class_num)

    non_masked_indices = np.where(~flat_sorted_masked.mask)
    non_masked_flat = flat_sorted[non_masked_indices]
    bnd = [non_masked_flat[i] for i in range(0, count, batch_sample)]
    bnd.append(non_masked_flat[-1] + 1e-10) # max boundary must be a bit higher than real max
    bnd[0] = bnd[0] - 1e-10 # min boundary must be a bit lower than real min
    bnd = np.array(bnd)

    # EFD_conversion
    thailand_class = np.empty(mjjaso_thailand.shape)
    for year in range(65):
        flat = mjjaso_thailand[year, :, :].reshape(-1)
        flat_masked = _mask(flat)
        is_masked = ma.getmask(flat_masked)
        grid_class = np.empty(len(flat))
        for i in range(lat_grid*lon_grid):
            if is_masked[i] == True:
                grid_class[i] = -99
            else:
                label = bisect.bisect(bnd, flat_masked[i])
                grid_class[i] = int(label - 1)
        grid_class = grid_class.reshape(lat_grid, lon_grid)
        thailand_class[year, :, :] = grid_class
    return thailand_class, bnd


if __name__ == '__main__':
    main()
    
