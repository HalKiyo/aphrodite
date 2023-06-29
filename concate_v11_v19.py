import netCDF4
import calendar
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def main():
    NOMASK = no_maske_concat()
    prcp = NOMASK.get_aphro1951()
    prcp = NOMASK.get_previous_aphro(prcp)
    prcp = NOMASK.get_latest_aphro(prcp)
    plt.imshow(prcp[-1, :, :])
    plt.show()

    # save data
    savepath  = "/work/kajiyama/preprocessed/APHRODITE/" \
                "v11_v19_monthly_mean_mm_per_day_1951-2015.npy"
    np.save(savepath, prcp)
    print(prcp.shape)

###############################################################################
#if basin mask is needed use below
##############################################################################
class mask_preprocess():
    def make_cprb_mask(self):
        """
        shape_size: [280(15S-55N, 360(60E-150E)]
        mask_value: 99999
        mask_size: lat20=20N, ln100=100E;
        resolution = 0.25 x 0.25 degree
        adding mask_value along longitude from lon100 - 10 to lon100 - 2
        """
        mask_value = 10**5
        white = np.zeros((280, 360))
        lt20 = (55 - 20)*4 - 1
        ln100 = (100 - 59)*4 - 1

        white[lt20+3 : lt20+8, ln100-10] = mask_value
        white[lt20+10 : lt20+12, ln100-10] = mask_value
        white[lt20+2 : lt20+12, ln100-9] = mask_value
        white[lt20+2 : lt20+13, ln100-8] = mask_value
        white[lt20+3 : lt20+18, ln100-7] = mask_value
        white[lt20+5 : lt20+21, ln100-6] = mask_value
        white[lt20+3 : lt20+25, ln100-5] = mask_value
        white[lt20+5 : lt20+25, ln100-4] = mask_value
        white[lt20+5 : lt20+27, ln100-3] = mask_value
        white[lt20+4 : lt20+27, ln100-2] = mask_value
        white[lt20+3 : lt20+27, ln100-1] = mask_value
        white[lt20+3 : lt20+25, ln100] = mask_value
        white[lt20+12 : lt20+23, ln100+1] = mask_value
        white[lt20+12 : lt20+21, ln100+2] = mask_value

        return white


###############################################################################
#if basin mask is not needed just run below
##############################################################################
class no_maske_concat():
    def __init__(self):
        self.top = 0,
        self.bottom= 280,
        self.left = 0,
        self.right = 360
        self.fill_value = -99
        self.base_path = "/work/kajiyama/data/APHRODITE/" \
                         "V1101_MA_025d/APHRO_MA_025deg_V1101.1951.nc"

    def original_mask(self, days=365, bottom=280, right=360):
        v11 = self.base_path
        nc = netCDF4.Dataset(v11, 'r')
        var = nc.variables['precip'][:]
        shape = np.reshape(var, (days, bottom, right))
        # mask of end of july
        reverse = shape[210, ::-1, :]
        aphro_mask = reverse.mask
        return aphro_mask

    def _fill(self, masked):
        filled = ma.filled(masked, fill_value=self.fill_value)
        return filled

    def _mask(self, filled):
        masked = ma.masked_where(filled <= self.fill_value, filled)
        return masked

    def open_netcdf(self, path, days, top=0, bottom=280, left=0, right=360):
        """
        reverse: numpy.ma.core.MaskedArray
        """
        nc = netCDF4.Dataset(path, 'r')
        var = nc.variables['precip'][:]
        shape = np.reshape(var, (days, bottom, right))
        reverse = shape[:, ::-1, :]
        reverse_masked = self._mask(reverse)
        return reverse_masked

    def get_aphro1951(self):
        """
        reverse.shape = (days, lat_num, lon_num)
        use mean instead of sum for monthly data creation
        -> monthly average mm/day
        -> because there might be missing data with month
        """
        v11 = self.base_path
        reverse_masked = self.open_netcdf(v11, 365)

        # monthly conversion
        ind = 0
        cal = calendar.monthrange(1951, 1)
        numdays = cal[1]
        # monthtotal is maskedarray
        monthtotal = np.mean(reverse_masked[ind : ind+numdays], axis=0)
        # monthtotal_filled is ndarray
        monthtotal_filled = self._fill(monthtotal)
        # prcp is ndarray:  mask_value will be automatically filled with 0 if maskedarray
        prcp = np.array([monthtotal_filled])

        ind += numdays
        for mon in range(2, 13):
            cal = calendar.monthrange(1951, mon)
            numdays = cal[1]
            monthtotal = np.mean(reverse_masked[ind:ind+numdays], axis=0)
            monthtotal_filled = self._fill(monthtotal)
            arry = np.array([monthtotal_filled])
            prcp = np.concatenate([prcp, arry])
            ind += numdays

        return prcp

    def get_previous_aphro(self, prcp):
        for year in range(1952, 1998):
            v11 = f"/work/kajiyama/data/APHRODITE/V1101_MA_025d/APHRO_MA_025deg_V1101.{year}.nc"
            if calendar.isleap(year):
                reverse_masked = self.open_netcdf(v11, 366)
            else:
                reverse_masked = self.open_netcdf(v11, 365)
            ind = 0
            for mon in range(1, 13):
                cal = calendar.monthrange(year, mon)
                numdays = cal[1]
                monthtotal = np.mean(reverse_masked[ind : ind+numdays], axis=0)
                monthtotal_filled = self._fill(monthtotal)
                arry = np.array([monthtotal_filled])
                prcp = np.concatenate([prcp, arry])
                ind += numdays

        return prcp

    def open_binary(self, path, days):
        raw = np.fromfile(path, 'float32').reshape(days, 2, 280, 360)
        reverse = raw[:, 0, ::-1, :]
        reverse_masked = self._mask(reverse)
        return reverse_masked

    def get_latest_aphro(self, prcp):
        for year in range(1998, 2016):
            v19 = f"/work/kajiyama/data/APHRODITE/V1901_MA_025d/APHRO_MA_025deg_V1901.{year}"
            if calendar.isleap(year):
                reverse_masked = self.open_binary(v19, 366)
            else:
                reverse_masked = self.open_binary(v19, 365)
            ind = 0
            for mon in range(1, 13):
                cal = calendar.monthrange(year, mon)
                numdays = cal[1]
                monthtotal = np.mean(reverse_masked[ind : ind+numdays], axis=0)
                monthtotal_filled = self._fill(monthtotal)
                arry = np.array([monthtotal_filled])
                prcp = np.concatenate([prcp, arry])
                ind += numdays

        return prcp


if __name__ == '__main__':
    main()
