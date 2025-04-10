
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
    return (scipy.interpolate.interp1d(energies, prob_spec), energies, prob_spec)


#modifies input size to change the spot size
def change_parameter(parameters, new_val, multiplier=1, new_input_filename='input'):
    new_val = new_val[0]
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

#returns likelihood given energies array and two probability spectra functions
def get_likelihood(energies, spec_exp, spec_comp):
    likelihood = 1
    for x in energies:
        pseudo_num_particles = spec_exp(x)
        likelihood = likelihood * np.power(spec_comp(x), pseudo_num_particles)
    return likelihood

def get_log_likelihood(energies, spec_exp, spec_comp):
    likelihood = 0
    for x in energies:
        likelihood = likelihood + (spec_exp(x) * np.log(spec_comp(x)))
    return likelihood






#mle settings
num_particles = 64
ext = '_e64p64_nmtest_fixed'
folder_number = 4
plot_type = 'spec'
save_ext = ''
cores = 64
parameters = np.array(['sigma_x', 'sigma_y'])
#parameters = np.array(['emit_n_x', 'emit_n_y'])
#parameters = np.array(['sigma_x', 'sigma_y'])
#parameter = 'emit'  #for file names
parameter = 'ss'
#param_units = ''
param_units = 'micron' #for file names
unit_multiplier = 1e-6
plot_axis = 'Beam $\sigma$ ($\mu m$)'
#plot_axis = 'Beam emittance ($\mu m$)'
#plot_axis = 'Beam ion charge state'

experimental_val = 1.0
initial_guess = 1.1
maxfev = 10
llh_arr = np.zeros(maxfev)
param_arr = np.zeros(maxfev)

#folder setup
for folder in (f'rad_data{folder_number}', f'outputs{folder_number}', f'results{folder_number}', f'mle{folder_number}'):
    if not os.path.exists(f'{folder}/{num_particles}/{parameter}{ext}'):
        os.makedirs(f'{folder}/{num_particles}/{parameter}{ext}')

#running experimental
'''
print(f'running experimental simulation for {parameter} {experimental_val}')
os.system('date +%c')
change_parameter(parameters, [experimental_val], multiplier=unit_multiplier)
os.system(f'mpirun -np {cores} model1 input')
'''
rad_exp = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{experimental_val}_{param_units}_exp'
output_exp = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{experimental_val}_{param_units}_exp'
#os.system(f'cp radiation {rad_exp}')
#os.system(f'cp output {output_exp}')
spec_exp, _, _ = get_spectrum_function(output_exp, rad_exp)

def llh_from_param_val(val, spec_exp=spec_exp, llh_arr=llh_arr, parameters=parameters, parameter='ss', param_units='micron', folder_number=4, num_particles=64):
    print(f'running simulation for {parameter} {val}')
    os.system('date +%c')
    change_parameter(parameters, val, multiplier=unit_multiplier)
    os.system(f'mpirun -np {cores} model1 input')
    os.system(f'cp radiation rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}')
    os.system(f'cp output outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}')
    if abs(val - experimental_val) < .001:
        print(f'running experimental simulation for {parameter} {val}')
        os.system('date +%c')
        os.system(f'mpirun -np {cores} model1 input')
        os.system(f'cp radiation {rad_exp}')
        os.system(f'cp output {output_exp}')
    print(f'{val} done at:')
    os.system('date +%c')

    general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}'
    radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}'
    spec_comp, energies, _ = get_spectrum_function(general_output_filename, radiation_output_filename)
    log_llh = get_log_likelihood(energies, spec_exp, spec_comp)
    i = 0
    while (i < len(llh_arr)-1) and (llh_arr[i] != 0):
        i += 1
    llh_arr[i] = log_llh
    param_arr[i] = val

    print(f'{parameter}, llh: {val}, {llh_arr[i]}')
    return -log_llh


def avg_llh_from_param_val(val, spec_exp=spec_exp, llh_arr=llh_arr, parameters=parameters, parameter='ss', param_units='micron', folder_number=4, num_particles=64):
    for n in range(3):ㅜ                      
        print(f'running simulation for {parameter} {val}')
        os.system('date +%c')
        change_parameter(parameters, val, multiplier=unit_multiplier)
        os.system(f'mpirun -np {cores} model1 input')
        os.system(f'cp radiation rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}')
        os.system(f'cp output outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}')
        if abs(val - experimental_val) < .001:
            print(f'running experimental simulation for {parameter} {val}')
            os.system('date +%c')
            os.system(f'mpirun -np {cores} model1 input')
            os.system(f'cp radiation {rad_exp}')
            os.system(f'cp output {output_exp}')
        print(f'{val} done at:')
        os.system('date +%c')

        general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{val}_{param_units}'
        radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{val}_{param_units}'
        spec_comp, energies, _ = get_spectrum_function(general_output_filename, radiation_output_filename)
        log_llh = get_log_likelihood(energies, spec_exp, spec_comp)

    i = 0
    while (i < len(llh_arr)-1) and (llh_arr[i] != 0):
        i += 1
    llh_arr[i] = log_llh
    param_arr[i] = val

    print(f'{parameter}, llh: {val}, {llh_arr[i]}')
    return -log_llh


#main

#default setup stuff
cores = 64
for folder in (f'rad_data{folder_number}', f'outputs{folder_number}', f'results{folder_number}'):
    if not os.path.exists(f'{folder}/{num_particles}/{parameter}{ext}'):
        os.makedirs(f'{folder}/{num_particles}/{parameter}{ext}')

#nelder mead maximization

opt_val = minimize(llh_from_param_val, initial_guess, method='Nelder-Mead', options={'maxfev':maxfev})
print(f'mle result:  {opt_val} {param_units}')
print(f'llh arr: {llh_arr}')
print(f'param arr: {param_arr}')

#making mle spec plot

#llh_arr = np.array([-2.75123643, -2.76226097, -2.77382686, -2.75232916, -2.76066378, -2.74934715, -2.75162347, -2.75057177, -2.74991617, -2.74943161])
#param_arr = np.array([1.1, 1.155, 1.21, 1.1825, 1.21, 1.16875, 1.155, 1.175625, 1.161875, 1.1721875])

if plot_type == 'spec' or plot_type == 'both':
    #organize experimental data/spectrum
    #spec_exp, energies, _, {param_units} = get_spectrum_function(output_exp, rad_exp, set_xlim=2, xlim=xlim)
    spec_exp, energies, _ = get_spectrum_function(f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_{experimental_val}_{param_units}_exp', f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_{experimental_val}_{param_units}_exp')


    #calculate likelihoods from spectrum (organize all other spectrums)
    for i, val in enumerate(param_arr):
        general_output_filename = f'outputs{folder_number}/{num_particles}/{parameter}{ext}/output_[{val}]_{param_units}'
        radiation_output_filename = f'rad_data{folder_number}/{num_particles}/{parameter}{ext}/rad_[{val}]_{param_units}'
        spec_comp, _, _ = get_spectrum_function(general_output_filename, radiation_output_filename)
        llh_arr[i] = get_log_likelihood(energies, spec_exp, spec_comp)
        print(f'{parameter}, llh: {val}, {llh_arr[i]}')

    #find peak
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot log-llhs
    plt.scatter(param_arr, llh_arr)
    plt.scatter(param_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
    plt.ylabel('Log Likelihood')
    plt.xlabel(plot_axis)
    plt.legend()

    plt.savefig(f'mle{folder_number}/{num_particles}/spec_{parameter}{ext}.png', dpi = 300)
    print("mle plot saved")
