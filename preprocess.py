import pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cmaps
from mpl_toolkits import basemap

def main():
    save_flag = True
    PRE = preprocess()
    DRA = draw_tools()

    # prcp: ndarray filled with -99
    prcp = PRE.load_npy(PRE.loadpath)
    pr_anom = PRE.anomaly(prcp)
    pr_clim, pr_variance, pr_std = PRE.standardize(prcp)
    pr_coarse = PRE.interpolation(prcp)
    pr_coarse_anom = PRE.anomaly(pr_coarse)
    pr_coarse_clim, pr_coarse_variance, pr_coarse_std = PRE.standardize(pr_coarse)
    PRE.save(prcp, pr_clim, pr_variance, pr_anom, pr_std,
             pr_coarse, pr_coarse_clim, pr_coarse_variance, pr_coarse_anom, pr_coarse_std,
             save_flag=save_flag)

class preprocess():
    def __init__(self):
        self.loadpath  = "/work/kajiyama/preprocessed/APHRODITE" \
                         "/v11_v19_monthly_mean_mm_per_day_1951-2015.npy"
        self.savepath = "/work/kajiyama/preprocessed/APHRODITE" \
                        "/aphrodite_anom_std_interp.pickle"
        self.mask_value = -99

    def _fill(self, x):
        f = ma.filled(x, fill_value=self.mask_value)
        return f

    def _mask(self, x):
        m = ma.masked_where(x<=self.mask_value, x)
        return m

    def load_npy(self, path):
        """
        prcp.shape = (780, 280, 360)
        """
        prcp = np.load(path)
        return prcp

    def anomaly(self, x):
        dup = x.copy()
        dup_masked = self._mask(dup)
        pr_anom = np.empty(dup.shape)
        for mon in range(12):
            clim_masked = np.mean(dup_masked[mon::12, :, :], axis=0)
            anom_masked = dup_masked[mon::12, :, :] - clim_masked
            anom_filled = self._fill(anom_masked)
            pr_anom[mon::12, :, :] = anom_filled
        return pr_anom

    def standardize(self, x):
        dup = x.copy()
        dup_masked = self._mask(dup)
        pr_std = np.empty(dup.shape)
        pr_clim = np.empty((12, dup.shape[1], dup.shape[2]))
        pr_variance = np.empty((12, dup.shape[1], dup.shape[2]))

        for mon in range(12):
            clim_masked = np.mean(dup_masked[mon::12, :, :], axis=0)
            clim_filled = self._fill(clim_masked)
            pr_clim[mon, :, :] = clim_filled

            variance_masked = np.std(dup_masked[mon::12, :, :], axis=0)
            variance_filled = self._fill(variance_masked)
            pr_variance[mon, :, :] = variance_filled

            std_masked = (dup_masked[mon::12, :, :] - clim_masked) / variance_masked
            std_filled = self._fill(std_masked)
            pr_std[mon::12, :, :] = std_filled

        return pr_clim, pr_variance, pr_std

    def interpolation(self, x):
        """
        S15-N55
        E60-E150
        """
        lt, ln = 280, 360 # grid number of latitude and longitude
        lons = np.linspace(60, 150, 360)
        lats = np.linspace(-15, 55, 280)
        xi = np.linspace(60, 150, 90)
        yi = np.linspace(-15, 55, 70)
        xi, yi = np.meshgrid(xi, yi)

        dup = x.copy()
        dup_masked = self._mask(dup)
        # return array has coarser shape
        pr_coarse = dup[:, :70, :90]


        for time in range(len(pr_coarse)):
            interp_masked = basemap.interp(dup_masked[time, :, :],
                                           lons, lats,
                                           xi, yi,
                                           order=0)
            interp_filled = self._fill(interp_masked)
            pr_coarse[time, :, :] = interp_filled

        return pr_coarse

    def  save(self, raw, pr_clim, pr_variance, pr_anom, pr_std,
              pr_coarse, pr_coarse_clim, pr_coarse_variance, pr_coarse_anom, pr_coarse_std,
              save_flag = False):
        save_dict = {"pr_raw": raw,
                     "pr_clim": pr_clim,
                     "pr_variance": pr_variance,
                     "pr_anom": pr_anom,
                     "pr_std": pr_std,
                     "pr_coarse": pr_coarse,
                     "pr_coarse_clim": pr_coarse_clim,
                     "pr_coarse_variance": pr_coarse_variance,
                     "pr_coarse_anom": pr_coarse_anom,
                     "pr_coarse_std": pr_coarse_std,
                     }

        print(f"pr_raw: {raw.shape}",
              f"pr_clim: {pr_clim.shape}",
              f"pr_variance: {pr_variance.shape}",
              f"pr_anom: {pr_anom.shape}",
              f"pr_std: {pr_std.shape}",
              f"pr_coarse: {pr_coarse.shape}",
              f"pr_coarse_clim: {pr_coarse_clim.shape}",
              f"pr_coarse_variance: {pr_coarse_variance.shape}",
              f"pr_coarse_anom: {pr_coarse_anom.shape}",
              f"pr_coarse_std: {pr_coarse_std.shape}",
             ) 

        if save_flag is True:
            with open(self.savepath, 'wb') as f:
                pickle.dump(save_dict, f)
            print('aphrodite pickle file has been SAVED')
        else:
            print('aphrodite pickle file has ***NOT*** been saved yet')



class draw_tools():
    def boxplot(self, prcp):
        df = pd.DataFrame({'Jan': prcp[0::12],
                           'Feb': prcp[1::12],
                           'Mar': prcp[2::12],
                           'Apr': prcp[3::12],
                           'May': prcp[4::12],
                           'Jun': prcp[5::12],
                           'Jul': prcp[6::12],
                           'Aug': prcp[7::12],
                           'Sep': prcp[8::12],
                           'Oct': prcp[9::12],
                           'Nov': prcp[10::12],
                           'Dec': prcp[11::12]})
        points = (df['Jan'],
                  df['Feb'],
                  df['Mar'],
                  df['Apr'],
                  df['May'],
                  df['Jun'],
                  df['Jul'],
                  df['Aug'],
                  df['Sep'],
                  df['Oct'],
                  df['Nov'],
                  df['Dec'])
        x = np.arange(1, 13)
        flood = [1995, 2006, 2011]
        target_year = 2011

        fig, ax = plt.subplots()

        bp = ax.boxplot(points, sym='')
        ax.set_xticklabels=(df.columns)

        for i in flood:
            year = 12*(i - 1951)
            if i == target_year:
                ax.plot(x, prcp[year : year+12], color='r')
            else:
                ax.plot(x, prcp[year : year+12], color='g')

        plt.xlabel('month', fontsize=15)
        plt.ylabel('Monthly rainfall (mm)', fontsize = 15)
        plt.show()

    def _imshow(self, image):
        """
        shape_size: [280(15S-55N, 360(60E-150E)]
        """
        plt.register_cmap('WhiteBlueGreenYellowRed', cmaps.WhiteBlueGreenYellowRed)
        cmap = plt.get_cmap('WhiteBlueGreenYellowRed', 10)

        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-120, -30, -15, 55)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin='upper',
                         extent=img_extent,
                         transform=projection,
                         #vmax=500,
                         #cmap=cmap
                         )
        cbar = fig.colorbar(mat,
                            ax=ax,
                            orientation='horizontal')
        plt.show()

    def _plot_anom(self, image):
        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-120, -30, -15, 55)
        cmap = plt.cm.get_cmap('BrBG', 10)

        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin='upper',
                         extent=img_extent,
                         transform=projection,
                         norm=colors.CenteredNorm(),
                         cmap=cmap,
                         )
        cbar = fig.colorbar(mat, ax=ax, orientation='horizontal')
        plt.show()

    def _plot_std(self, image):
        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-120, -30, -15, 55)
        cmap = plt.cm.get_cmap('BrBG', 10)

        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin='upper',
                         extent=img_extent,
                         transform=projection,
                         norm=colors.Normalize(vmin=-3, vmax=3),
                         cmap=cmap,
                         )
        cbar = fig.colorbar(mat, ax=ax, orientation='horizontal')
        plt.show()


if __name__ == '__main__':
    main()
