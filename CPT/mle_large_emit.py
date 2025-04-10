import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate
import os
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy import signal

def linspace_midpoint(min, max, num):
    #returns a linspace array w/ given given start/stop (num is num elements in array)
    #returns a linspace array with half a step added to front and end, in midpoints of first array
    #returns the step size (same for both)
    vals, step = np.linspace(min, max, num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((min - 0.5 * step), (max + 0.5 * step), (num + 1), endpoint=True)
    return (vals, vals_midpoint, step)

def get_log_likelihood(energies, spec_exp, spec_comp):
    likelihood = 0
    for x, energy in enumerate(energies):
        likelihood = likelihood + (spec_exp[x] * np.log(spec_comp[x]))
    return likelihood


#main

#mle settings
data_file = 'training/training_data_emit.csv'
ss_file = 'training/training_emit.csv'
test_file = 'training/test_data_emit.csv'
num_particles = 64
ext = '_e64p64_train'
run_sim = True
folder_number = 4
plot_type = 'spec'
parameters = np.array(['emit_n_x', 'emit_n_y'])
parameter = 'emit'
param_units = 'micron' #for file names
unit_multiplier = 1e-6
plot_axis = 'Beam $\sigma$ ($\mu m$)'
cores = 64

energies = [31746.031746031746, 63492.06349206349, 95238.09523809524, 126984.12698412698, 158730.15873015873, 190476.19047619047, 222222.22222222222, 253968.25396825396, 285714.2857142857, 317460.31746031746, 349206.34920634923, 380952.38095238095, 412698.41269841266, 444444.44444444444, 476190.4761904762, 507936.50793650793, 539682.5396825396, 571428.5714285714, 603174.6031746032, 634920.6349206349, 666666.6666666666, 698412.6984126985, 730158.7301587302, 761904.7619047619, 793650.7936507936, 825396.8253968253, 857142.8571428572, 888888.8888888889, 920634.9206349206, 952380.9523809524, 984126.9841269841, 1015873.0158730159, 1047619.0476190476, 1079365.0793650793, 1111111.111111111, 1142857.1428571427, 1174603.1746031747, 1206349.2063492064, 1238095.238095238, 1269841.2698412698, 1301587.3015873015, 1333333.3333333333, 1365079.365079365, 1396825.396825397, 1428571.4285714286, 1460317.4603174604, 1492063.492063492, 1523809.5238095238, 1555555.5555555555, 1587301.5873015872, 1619047.619047619, 1650793.6507936507, 1682539.6825396826, 1714285.7142857143, 1746031.746031746, 1777777.7777777778, 1809523.8095238095, 1841269.8412698412, 1873015.873015873, 1904761.9047619049, 1936507.9365079366, 1968253.9682539683, 2000000.0]

spec_data = np.loadtxt(data_file, delimiter=',')
test_data = np.loadtxt(test_file, delimiter=',')
predictions = np.empty(len(test_data))
test_ss_arr = np.empty(len(test_data))

for j, case in enumerate(test_data):
    param_arr = np.empty(len(spec_data))
    llh_arr = np.empty(len(param_arr))

    # set up test case
    test_ss_arr[j] = case[0]
    test_spec = case[1:]
    test_spec_norm = [x / sum(test_spec) for x in test_spec]

    for i, line in enumerate(spec_data):
        param_arr[i] = line[0]
        spec = line[1:]
        spec_norm = [x / sum(spec) for x in spec]
        llh_arr[i] = get_log_likelihood(energies, test_spec_norm, spec_norm)

    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    firstpeak = peak[0][0]
    predictions[j] = param_arr[firstpeak]

plt.scatter(test_ss_arr, predictions, label='actual')
plt.plot(test_ss_arr, test_ss_arr, color='gray', alpha=.5, label='expected')
plt.xlabel('Actual Emittance ($\mu$m)')
plt.ylabel('Predicted Emittance ($\mu$m)')
plt.title('MLE-Predicted Emittances from 1D Energy Spectra')
plt.legend()
plt.savefig(f'mle{folder_number}/{num_particles}/spec_{parameter}{ext}_bigmle_emit.png', dpi = 300)
print("mle plot saved")
print()

# calculate mean squared error
mse = (np.square(predictions - test_ss_arr)).mean()
print(mse)

#making mle spec plot
'''
if plot_type == 'spec' or plot_type == 'both':
    #find peak
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot log-llhs
    s = param_arr.argsort()
    plt.plot(param_arr[s], llh_arr[s])
    plt.scatter(param_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
    plt.ylabel('Log Likelihood')
    plt.xlabel(plot_axis)
    plt.legend()

    plt.savefig(f'mle{folder_number}/{num_particles}/spec_{parameter}{ext}_{test_ss}.png', dpi = 300)
    print("mle plot saved")
    print(f'peak: {param_arr[peak]} microns')
'''
