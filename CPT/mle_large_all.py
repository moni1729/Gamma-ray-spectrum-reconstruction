import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate
import os
from scipy.signal import find_peaks
from scipy.optimize import minimize

def linspace_midpoint(min, max, num):
    vals, step = np.linspace(min, max, num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((min - 0.5 * step), (max + 0.5 * step), (num + 1), endpoint=True)
    return (vals, vals_midpoint, step)

def logspace_midpoint(min, max, num):
    vals, step = np.linspace(np.log(min), np.log(max), num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((np.log(min) - 0.5 * step), (np.log(max) + 0.5 * step), (num + 1), endpoint=True)
    return np.exp(vals), np.exp(vals_midpoint), None

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


#gives probability distribution function in phix/energy
def get_dd_function(output_filename, rad_filename):
    output_dict = {}
    with open(output_filename, 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])
    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    energies2, _, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])

    result_wit = np.fromfile(rad_filename).reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))
    dd2_wit = (np.sum(dd_wit, axis=2) * phi_ys_step).T
    dd_prob = dd2_wit / dd2_wit.sum()
    dd_fun = scipy.interpolate.interp2d(energies, phi_xs, dd_prob)
    return(dd_fun, energies, phi_xs)

# gives a probability distribution function in phiy/phix
def get_dist_function(output_filename, rad_filename):
    output_dict = {}
    with open(output_filename, 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])
    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    energies2, _, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])

    result_wit = np.fromfile(rad_filename).reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))
    dist = np.sum(0.5 * (dd_wit[1:, :, :] + dd_wit[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
    dist_prob = dist / dist.sum()
    dist_fun = scipy.interpolate.interp2d(phi_xs, phi_ys, dist_prob)
    return(dist_fun, phi_xs, phi_ys, dist_prob)

def get_2d_likelihood(x_arr, y_arr, fun_exp, fun_comp, total_particles=32):
    likelihood = 1
    #print(f'initial likelihood: {likelihood}')
    for x in x_arr:
        for y in y_arr:
            pseudo_num_particles = fun_exp(x, y) #* total_particles
            #print(f'likelihood function value: {fun_comp(x, y)}')
            likelihood = likelihood * np.power(fun_comp(x, y), pseudo_num_particles)
            #print(likelihood)
    return likelihood

def get_log_likelihood(energies, spec_exp, spec_comp):
    likelihood = 0
    for x in energies:
        likelihood = likelihood + (spec_exp(x) * np.log(spec_comp(x)))
    return likelihood

def get_2d_likelihood_twist(x_arr, y_arr, fun_exp, dist_comp, total_particles=32):
    num_positions = 4
    max_llh = -1
    for i in range(num_positions):
        fun_comp = scipy.interpolate.interp2d(x_arr, y_arr, dist_comp)
        new_llh = 1
        for x in x_arr:
            for y in y_arr:
                pseudo_num_particles = fun_exp(x, y) * total_particles
                new_llh = new_llh * np.power(fun_comp(x, y), pseudo_num_particles)
        if i==0 or new_llh > max_llh:
            max_llh = new_llh
        dist_rot = np.ones_like(dist)
        for x in range(len(dist[0, :])):
            for y in range(len(dist[:, 0])):
                dist_rot[-y][x] = dist[x][y]
        dist_comp = dist_rot

    return max_llh


#mle settings
data_file = 'training/training_data.csv'
ss_file = 'training/training_ss.csv'
test_file = 'training/test_data_2.csv'
num_particles = 64
ext = '_e64p64_train'
run_sim = True
folder_number = 4
spectrum_type = 'spec'
parameters = np.array(['sigma_x', 'sigma_y'])
parameter = 'ss'
param_units = 'micron' #for file names
unit_multiplier = 1e-6
plot_axis = 'Beam $\sigma$ ($\mu m$)'
cores = 64

train_data = np.loadtxt(data_file, delimiter=',')
ss_arr = np.sort(train_data[:,0])

llh_arr = np.empty(len(ss_arr))

output_file = 'outputs4/64/ss_e64p64_train/output_2.2062168899388865_micron'
rad_file = 'rad_data4/64/ss_e64p64_train/rad_2.2062168899388865_micron'

if spectrum_type == "spec":
    spec_exp, energies, _ = get_spectrum_function(output_file, rad_file)

    #calculate likelihoods from spectrum (organize all other spectrums)
    for i, val in enumerate(ss_arr):
        spec_comp, _, _ = get_spectrum_function(f'outputs4/64/ss_e64p64_train/output_{val}_micron', f'rad_data4/64/ss_e64p64_train/rad_{val}_micron')
        llh_arr[i] = get_log_likelihood(energies, spec_exp, spec_comp)
        print(f'ss, llh: {val}, {llh_arr[i]}')

    #find peak
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot log-llhs
    plt.scatter(ss_arr, llh_arr)
    plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
    plt.ylabel('Log Likelihood')
    plt.xlabel(plot_axis)
    plt.legend()
    plt.savefig(f'results7/ipac_specmle.png', dpi = 400)
    print("saved")
    plt.clf()

if spectrum_type == "dd" or spectrum_type == "both":
    dd_exp, energies, phi_xs = get_dd_function(output_file, rad_file)
    #calculate likelihoods from dd plot
    for i, spot_size in enumerate(ss_arr):
        dd_fun, energies_comp, phi_xs_comp = get_dd_function(f'outputs4/64/ss_e64p64_train/output_{spot_size}_micron', f'rad_data4/64/ss_e64p64_train/rad_{spot_size}_micron')
        llh_arr[i] = get_2d_likelihood(energies, phi_xs, dd_exp, dd_fun, total_particles=num_particles)
        print(f'ss, llh: {spot_size}, {llh_arr[i]}')
    #find peak
    print(ss_arr)
    print(llh_arr)
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot dd llhs
    plt.plot(ss_arr, llh_arr)
    plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
    plt.xlabel('Beam $\sigma$ ($\mu m$)')
    plt.ylabel('Likelihood')
    plt.legend()
    #plt.savefig(f'results7/ipac_ddmle.png', dpi = 400)
    print("saved")
    plt.clf()

if spectrum_type == "dist" or spectrum_type == "both":
    dist_exp, phi_xs, phi_ys, _ = get_dist_function(output_file, rad_file)
    #calculate likelihoods from dist plot
    for i, spot_size in enumerate(ss_arr):
        dist_fun, _, _, dist = get_dist_function(f'outputs4/64/ss_e64p64_train/output_{spot_size}_micron', f'rad_data4/64/ss_e64p64_train/rad_{spot_size}_micron')
        #llh_arr[i] = get_2d_likelihood_twist(phi_xs, phi_ys, dist_exp, dist)
        llh_arr[i] = get_2d_likelihood(phi_xs, phi_ys, dist_exp, dist_fun, total_particles=num_particles)
        print(f'ss, llh: {spot_size}, {llh_arr[i]}')
    #find peak
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot dist likelihoods
    plt.plot(ss_arr, llh_arr)
    plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)

    plt.xlabel('Beam $\sigma$ ($\mu m$)')
    plt.ylabel('Likelihood')
    plt.legend()
    #plt.savefig(f'results7/ipac_distmle.png', dpi = 400)
    print("saved")
    plt.clf()
