import csv
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

def get_dist_function(general_output_filename, radiation_output_filename):
    output_dict = {}
    with open(general_output_filename, 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])
    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    energies2, _, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])


    result_wit = np.fromfile(radiation_output_filename).reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))
    #dd2_wit = (np.sum(dd_wit, axis=2) * phi_ys_step).T
    #dd_prob = dd2_wit / dd2_wit.sum()
    dist = np.sum(0.5 * (dd_wit[1:, :, :] + dd_wit[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
    return(dist.tolist(), phi_xs_midpoint.tolist(), phi_ys_midpoint.tolist())


#modifies input size to change the spot size
def change_parameter(parameters, new_val, multiplier=1, new_input_filename='input'):
    input_new = open(new_input_filename, 'w')
    with open('input_template', 'r') as f:
        for line in f:
            split = line.split()
            is_param = False
            for parameter in parameters:
                if (split[0] == parameter):
                    input_new.write(line[:(len(parameter) + 3)] + str(new_val * multiplier) + '\n')
                    is_param = True
            if not is_param:
                input_new.write(line)


#settings
type = 'test' # training or test
num_ext = '_2' # include the _
data_file = f'training/{type}_data_dist{num_ext}.csv'
param_file = f'training/{type}_ss{num_ext}.csv'
ref_file = f'training/{type}_data{num_ext}.csv'
num_particles = 64
ext = '_e64p64_train'
folder_number = 4
parameters = np.array(['sigma_x', 'sigma_y'])
parameter = 'ss'
param_units = 'micron'
unit_multiplier = 1e-6

#default setup stuff
cores = 64
for folder in (f'rad_data{folder_number}', f'outputs{folder_number}', f'results{folder_number}'):
    if not os.path.exists(f'{folder}/{num_particles}/{parameter}{ext}'):
        os.makedirs(f'{folder}/{num_particles}/{parameter}{ext}')


#read and write to file
spec_data = np.loadtxt(ref_file, delimiter=',')


param_list = spec_data[:, 0]

with open(param_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(param_list)
# print(param_list)

# clear dist file first
with open(data_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow('')

for i, val in enumerate(param_list):
    general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}'
    radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}'
    dist, energies, phi_xs = get_dist_function(general_output_filename, radiation_output_filename)

    print(f'writing parameter value {i + 1}: {dist[0][0]}')
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        for row in dist:
            writer.writerow(row)

    if(i==0):
        print(f'energies: {energies}')
        print(f'phi_xs: {phi_xs}')
        print(f'shape of dist: {len(dist)}, {len(dist[0])}')
