import netCDF4
import calendar
import numpy as np
import matplotlib.pyplot as plt

from aphro_common import MASK_VALUE, fill, mask

# daily input (V11 netCDF for 1951-1997, V19 flat binary for 1998-2015)
V11_PATH = "/work/kajiyama/data/APHRODITE/V1101_MA_025d/APHRO_MA_025deg_V1101.{year}.nc"
V19_PATH = "/work/kajiyama/data/APHRODITE/V1901_MA_025d/APHRO_MA_025deg_V1901.{year}"
V11_YEARS = range(1952, 1998)  # 1951 is read separately by get_aphro1951
V19_YEARS = range(1998, 2016)
# monthly output
SAVEPATH = "/work/kajiyama/preprocessed/APHRODITE/" \
           "v11_v19_monthly_mean_mm_per_day_1951-2015.npy"

def main():
    NOMASK = no_maske_concat()
    prcp = NOMASK.get_aphro1951()
    prcp = NOMASK.get_previous_aphro(prcp)
    prcp = NOMASK.get_latest_aphro(prcp)
    plt.imshow(prcp[-1, :, :])
    plt.show()

    # save data
    np.save(SAVEPATH, prcp)
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
        self.fill_value = MASK_VALUE
        self.base_path = V11_PATH.format(year=1951)

    def original_mask(self, days=365, bottom=280, right=360):
        v11 = self.base_path
        with netCDF4.Dataset(v11, 'r') as nc:
            var = nc.variables['precip'][:]
        shape = np.reshape(var, (days, bottom, right))
        # mask of end of july
        reverse = shape[210, ::-1, :]
        aphro_mask = reverse.mask
        return aphro_mask

    def _fill(self, masked):
        return fill(masked, mask_value=self.fill_value)

    def _mask(self, filled):
        return mask(filled, mask_value=self.fill_value)

    def open_netcdf(self, path, days, top=0, bottom=280, left=0, right=360):
        """
        reverse: numpy.ma.core.MaskedArray
        """
        with netCDF4.Dataset(path, 'r') as nc:
            var = nc.variables['precip'][:]
        shape = np.reshape(var, (days, bottom, right))
        reverse = shape[:, ::-1, :]
        reverse_masked = self._mask(reverse)
        return reverse_masked

    def open_binary(self, path, days):
        raw = np.fromfile(path, 'float32').reshape(days, 2, 280, 360)
        reverse = raw[:, 0, ::-1, :]
        reverse_masked = self._mask(reverse)
        return reverse_masked

    def monthly_mean(self, reverse_masked, year):
        """
        Daily masked array of one year to a list of 12 monthly mean fields.
        use mean instead of sum for monthly data creation
        -> monthly average mm/day
        -> because there might be missing data with month
        Each element is an ndarray whose masked cells are filled with fill_value.
        """
        months = []
        ind = 0
        for mon in range(1, 13):
            numdays = calendar.monthrange(year, mon)[1]
            # monthtotal is maskedarray
            monthtotal = np.mean(reverse_masked[ind : ind+numdays], axis=0)
            # monthtotal_filled is ndarray
            months.append(self._fill(monthtotal))
            ind += numdays
        return months

    def _days(self, year):
        return 366 if calendar.isleap(year) else 365

    def get_aphro1951(self):
        """
        reverse.shape = (days, lat_num, lon_num)
        returns prcp with shape (12, lat_num, lon_num)
        """
        reverse_masked = self.open_netcdf(self.base_path, 365)
        return np.stack(self.monthly_mean(reverse_masked, 1951))

    def get_previous_aphro(self, prcp):
        months = []
        for year in V11_YEARS:
            reverse_masked = self.open_netcdf(V11_PATH.format(year=year), self._days(year))
            months.extend(self.monthly_mean(reverse_masked, year))
        return np.concatenate([prcp, np.stack(months)])

    def get_latest_aphro(self, prcp):
        months = []
        for year in V19_YEARS:
            reverse_masked = self.open_binary(V19_PATH.format(year=year), self._days(year))
            months.extend(self.monthly_mean(reverse_masked, year))
        return np.concatenate([prcp, np.stack(months)])


if __name__ == '__main__':
    main()
