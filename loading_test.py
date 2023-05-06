import utils.dl_utils as dl


data1 = dl.load_fits('/ibiscostorage/mdelliveneri/load_test_cube.fits')
data2 = dl.load_fits('/ibiscostorage/mdelliveneri/big_cube/clean_cube_0.fits')
print(data1.shape, data2.shape)