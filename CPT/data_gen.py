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

#returns spectrum as interpolated function and energy array
#also added the probability values as return in order to record spectra
def get_spectrum_function(general_output_filename, radiation_output_filename):
    #makes dictionary with parameters in the general output file
    output_dict = {}
    with open(general_output_filename, 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])
    #makes different arrays for differenct parameters-- focus on the energy
    #arrays based on the output parameters
    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    energies2, _, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])

    #makes spectrum arrays
    result_wit = np.fromfile(radiation_output_filename).reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)    #has length of energy_num, array of total energies
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))  #particles is num particles
    spectrum_wit = np.sum(dd_wit, axis=(1,2)) * phi_xs_step * phi_ys_step
    prob_spec = spectrum_wit / spectrum_wit.sum()

    #return spectrum as interpolated function and energy array
    return (spectrum_wit.tolist(), energies.tolist(), prob_spec)


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


#settings       # completed 24 of 30 last run
data_file = 'training/training_data.csv'
ss_file = 'training/training_ss.csv'
min_ss = .1
max_ss = 1.0
num_cases = 2

num_particles = 64
ext = '_e64p64_train'
folder_number = 4
plot_type = 'spec'
parameters = np.array(['sigma_x', 'sigma_y'])
parameter = 'ss'
param_units = 'micron'
unit_multiplier = 1e-6
plot_axis = 'Beam $\sigma$ ($\mu m$)'

'''
########################################
#########   range of ss values  ########
########################################
min_ss = 41
max_ss = min_ss + num_cases
param_file = 'training/training_ss.csv'

spec_data = np.loadtxt('training/training_data.csv', delimiter=',')
param_list = spec_data[:, 0]

with open(param_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(param_list)

read_list = np.loadtxt(param_file, delimiter=',')
ss_list = read_list[min_ss:max_ss]
'''

#default setup stuff
cores = 64
for folder in (f'rad_data{folder_number}', f'outputs{folder_number}', f'results{folder_number}'):
    if not os.path.exists(f'{folder}/{num_particles}/{parameter}{ext}'):
        os.makedirs(f'{folder}/{num_particles}/{parameter}{ext}')

ss_list = []
#running simulations
for i in range(num_cases):
    val = np.random.uniform(min_ss, max_ss)
    ss_list.append(val)

    print(f'running simulation {i+1}/{num_cases} for {parameter} {val}')
    os.system('date +%c')
    change_parameter(parameters, val, multiplier=unit_multiplier)
    os.system(f'mpirun -np {cores} model1 input')

    general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}'
    radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}'
    os.system(f'cp radiation {radiation_output_filename}')
    os.system(f'cp output {general_output_filename}')
    print(f'{val} done at:')
    os.system('date +%c')

    spec_comp, energies, _ = get_spectrum_function(general_output_filename, radiation_output_filename)
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        output = [val]
        output.extend(spec_comp)
        writer.writerow(output)
    print("energies: ")
    print(energies)

print(ss_list)
with open(ss_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(ss_list)

'''
list = [2.9185589690913285, 1.3130887821547759, 2.6056999831773524, 2.8431263426992452, 0.5472805506971629, 2.0877217414641107, 1.284018908298652, 1.0128935644680772, 0.30467830541987756, 0.9412012401255472]
for val in list:
    general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}'
    radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}'

    spec_comp, energies, _ = get_spectrum_function(general_output_filename, radiation_output_filename)
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        output = [val]
        output.extend(spec_comp)
        writer.writerow(output)

val = 0.12707896813273237
general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}'
radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}'
spec_comp, energies, _ = get_spectrum_function(general_output_filename, radiation_output_filename)
print(energies)
'''
