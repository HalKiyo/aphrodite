import numpy as np
import numpy.ma as ma

from aphro_common import MASK_VALUE, mask

WORKDIR = "/work/kajiyama/cnn/transfer_input/pr"

def main(class_num=5, discrete_mode='EFD', resolution='1x1'):
    save_flag = False
    if resolution == '1x1':
        lat_grid, lon_grid = 20, 20
        key = '1x1_coarse'

    fname = f"aphro_{key}_std_MJJASO_thailand"
    thailand_path = WORKDIR + f"/continuous/thailand/{resolution}/{fname}.npy"
    thailand_spath = WORKDIR + f"/class/thailand/{discrete_mode}" \
                     f"/{fname}_{discrete_mode}_{class_num}.npy"

    thailand = np.load(thailand_path) # thailand.shape=(65, 20, 20)

    if discrete_mode == 'EFD':
        thailand_class, thailand_bnd = thailand_EFD(thailand, class_num=class_num,
                                                    lat_grid=lat_grid, lon_grid=lon_grid)
        print(f"thailand_bnd: {thailand_bnd}")
        save_npy(thailand_spath, thailand_class, save_flag=save_flag)

def save_npy(path, data, save_flag=False):
    if save_flag is True:
        np.save(path, data)
        print("class_outut has been SAVED")
    else:
        print("class_output is ***NOT*** saved yet")

def thailand_EFD(data, class_num=5, lat_grid=20, lon_grid=20):
    # EFD_bnd
    mjjaso_thailand = data.copy() # data=(65, 20, 20)
    thailand_flat = mjjaso_thailand.reshape(-1)
    flat_sorted = np.sort(thailand_flat)
    flat_sorted_masked = mask(flat_sorted)
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
    # label = bisect.bisect(bnd, value) - 1 for every unmasked grid,
    # computed for all grids of one year at once with np.searchsorted(side='right')
    thailand_class = np.empty(mjjaso_thailand.shape)
    for year in range(65):
        flat = mjjaso_thailand[year, :, :].reshape(-1)
        is_masked = ma.getmaskarray(mask(flat))
        label = np.searchsorted(bnd, flat, side='right') - 1
        grid_class = np.where(is_masked, MASK_VALUE, label).astype(float)
        thailand_class[year, :, :] = grid_class.reshape(lat_grid, lon_grid)
    return thailand_class, bnd


if __name__ == '__main__':
    main()
