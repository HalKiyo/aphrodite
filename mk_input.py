import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def main():
    save_flag = False
    if save_flag is True:
        pr_1x1()

def _fill(x, mask_value=-99):
    f = ma.filled(x, fill_value=mask_value)
    return f

def _mask(x, mask_value=-99):
    m = ma.masked_where(x<=mask_value, x)
    return m

def lead_cnd(leadtime):
    if leadtime == 1:
        mon_arr = np.array([[4,5], [5,6], [6,7], [7,8], [8,9], [9,10]])
        llst = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    elif leadtime == 3:
        mon_arr = np.array([[4,7], [5,8], [6,9], [7,10]])
        llst = ['MJJ', 'JJA', 'JAS', 'ASO']
    elif leadtime == 6:
        mon_arr = np.array([[4,10]])
        llst = ['MJJASO']
    else:
        print('error of lead_cnd')
        exit()
    return mon_arr, llst

def extract_field(prcp, resolution='1deg'):
    """
    shape_size: [280(15S-55N, 360(60E-150E)]
    N5-25, E90-110
    0=54.75-55
    1=54.55-54.75
    2=54.25-54.55
    3=54-54.25
    """
    if resolution == '0.25deg':
        north = (55-25)*4 # N25
        south = north + (20*4) #N5
        west = (90-60)*4 #E90
        east = west + (20*4) # E110
        field = prcp[:, north:south, west:east]
    elif resolution == '1deg':
        north = (55-25) # N25
        south = north + (20) #N5
        west = (90-60) #E90
        east = west + (20) # E110
        field = prcp[:, north:south, west:east]

    return field

def pr_1x1():
    workdir = '/work/kajiyama/cnn/transfer_input/pr'

    variable = 'aphro'
    resolution = '1x1'
    form_lst = ['coarse', 'coarse_anom', 'coarse_std']
    space = 'thailand'
    lead_lst = [1, 3, 6]

    for form in form_lst:
        ifile = f"{workdir}/main/{variable}_{form}.npy"
        loaded = np.load(ifile)
        print(f"{ifile} is loaded")

        for leadtime in lead_lst:
            mon_arr, llst = lead_cnd(leadtime)
            for i, mon in enumerate(llst):

                thailand = extract_field(loaded, resolution='1deg')
                years = thailand.reshape(int(thailand.shape[0]/12),
                                         12,
                                         thailand.shape[1],
                                         thailand.shape[2])
                months = years[:, mon_arr[i,0] : mon_arr[i,1], :, :]
                months_masked = _mask(months)
                months_mean = np.mean(months_masked, axis=1)
                months_mean_filled = _fill(months_mean)

                outdir = f"{workdir}/continuous/{space}/{resolution}"
                outfile = f"{outdir}/{variable}_{resolution}_{form}_{mon}_{space}.npy"
                np.save(outfile, months_mean_filled)
                print(f"{outfile} is saved {months_mean_filled.shape}")


if __name__ == '__main__':
    main()
