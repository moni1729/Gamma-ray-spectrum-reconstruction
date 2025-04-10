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

#modifies input size to change the spot size
def change_spot_size(new_size, new_input_filename='input'):
    input_new = open(new_input_filename, 'w')
    with open('input_template', 'r') as f:
        for line in f:
            split = line.split()
            if(split[0] == 'sigma_x' or split[0] == 'sigma_y'):
                input_new.write(line[:10] + str(new_size * 1e-6) + '\n')
            else:
                input_new.write(line)

#gives probability distribution function in phix/energy
def get_dd_function(spot_size, is_experimental=False, num_part_name=64, name_extension=''):
    exp = ''
    if is_experimental:
        exp = '_exp'
    ext = exp + name_extension
    output_dict = {}
    with open(f'outputs2/{num_part_name}/output_{spot_size}_micron{ext}', 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])
    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    energies2, _, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])
    #print(len(energies), len(phi_xs), particles)
    #print(f'outputs2/{num_part_name}/output_{spot_size}_micron{exp}')

    result_wit = np.fromfile(f'rad_data2/{num_part_name}/rad_{spot_size}_micron{ext}').reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))
    dd2_wit = (np.sum(dd_wit, axis=2) * phi_ys_step).T
    dd_prob = dd2_wit / dd2_wit.sum()
    dd_fun = scipy.interpolate.interp2d(energies, phi_xs, dd_prob)
    return(dd_fun, energies, phi_xs)

# gives a probability distribution function in phiy/phix
def get_dist_function(spot_size, is_experimental=False, name_extension='', num_part_name=64):
    exp = ''
    if is_experimental:
        exp = '_exp'
    ext = exp + name_extension
    output_dict = {}
    with open(f'outputs2/{num_part_name}/output_{spot_size}_micron{ext}', 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])
    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    energies2, _, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])

    result_wit = np.fromfile(f'rad_data2/{num_part_name}/rad_{spot_size}_micron{ext}').reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))
    dist = np.sum(0.5 * (dd_wit[1:, :, :] + dd_wit[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
    dist_prob = dist / dist.sum()
    dist_fun = scipy.interpolate.interp2d(phi_xs, phi_ys, dist_prob)
    return(dist_fun, phi_xs, phi_ys, dist_prob)

def get_2d_likelihood(x_arr, y_arr, fun_exp, fun_comp, total_particles=32):
    likelihood = 1
    #print(f'initial likelihood: {likelihood}')
    print('num particles: ' + str(total_particles))
    for x in x_arr:
        for y in y_arr:
            pseudo_num_particles = fun_exp(x, y) #* total_particles
            #print(f'likelihood function value: {fun_comp(x, y)}')
            likelihood = likelihood * np.power(fun_comp(x, y), pseudo_num_particles)
            #print(likelihood)
    print(f'final likelihood: {likelihood}')
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
ss_arr = np.linspace(1.5, 2.5, 11)
#ss_arr = np.linspace(1, 3, 5)
llh_arr = np.empty(len(ss_arr))
experimental_ss = 2.0
num_particles = 250
ext = '_p50e25'
spectrum_type = "both"
cores = 64

#plot settings
set_ylim = False
y_min = .01
y_max = .039

#running simulations
for spot_size in ss_arr:
    print(f'running simulation for spot size {spot_size} μm')
    os.system('date +%c')
    change_spot_size(spot_size)
    os.system(f'mpirun -np {cores} model1 input')
    os.system(f'cp radiation rad_data2/{num_particles}/rad_{spot_size}_micron{ext}')
    os.system(f'cp output outputs2/{num_particles}/output_{spot_size}_micron{ext}')
    if abs(spot_size - experimental_ss) < .001:
        print(f'running experimental simulation for spot size {spot_size} μm')
        os.system('date +%c')
        os.system(f'mpirun -np {cores} model1 input')
        os.system(f'cp radiation rad_data2/{num_particles}/rad_{spot_size}_micron_exp{ext}')
        os.system(f'cp output outputs2/{num_particles}/output_{spot_size}_micron_exp{ext}')
    print(f'{spot_size} μm done at:')
    os.system('date +%c')
    #plot_from_files(spot_size)

if spectrum_type == "dd" or spectrum_type == "both":
    dd_exp, energies, phi_xs = get_dd_function(experimental_ss, is_experimental=True, num_part_name=num_particles, name_extension=ext)
    #calculate likelihoods from dd plot
    for i, spot_size in enumerate(ss_arr):
        dd_fun, energies_comp, phi_xs_comp = get_dd_function(spot_size, num_part_name=num_particles, name_extension=ext)
        #assert len(energies_comp) == len(energies), "energy array mismatch"
        #assert len(phi_xs_comp) == len(phi_xs), "phi_xs array mismatch"
        llh_arr[i] = get_2d_likelihood(energies, phi_xs, dd_exp, dd_fun, total_particles=num_particles)
        print(f'ss, llh: {spot_size}, {llh_arr[i]}')
    #find peak
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot dd llhs
    plt.plot(ss_arr, llh_arr)
    plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
    plt.xlabel('Beam $\sigma$ ($\mu m$)')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.savefig(f'mle2/{num_particles}/dd{ext}.png', dpi = 400)
    print("saved")
    plt.clf()


if spectrum_type == "dist" or spectrum_type == "both":
    dist_exp, phi_xs, phi_ys, _ = get_dist_function(experimental_ss, is_experimental=True, name_extension=ext, num_part_name=num_particles)
    #calculate likelihoods from dist plot
    for i, spot_size in enumerate(ss_arr):
        dist_fun, _, _, dist = get_dist_function(spot_size, num_part_name=num_particles, name_extension=ext)
        #llh_arr[i] = get_2d_likelihood_twist(phi_xs, phi_ys, dist_exp, dist)
        llh_arr[i] = get_2d_likelihood(phi_xs, phi_ys, dist_exp, dist_fun, total_particles=num_particles)
        print(f'ss, llh: {spot_size}, {llh_arr[i]}')
    #find peak
    max_llh = np.amax(llh_arr)
    peak = np.where(llh_arr == max_llh)
    #plot dist likelihoods
    plt.plot(ss_arr, llh_arr)
    plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
    if set_ylim:
        plt.ylim([y_min, y_max])
    plt.xlabel('Beam $\sigma$ ($\mu m$)')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.savefig(f'mle2/{num_particles}/dist{ext}.png', dpi = 400)
    print("saved")
    plt.clf()


#normalize llh and find peak
'''
area = np.trapz(llh_arr, x=ss_arr)
#llh_arr = llh_arr / area
max_llh = np.amax(llh_arr)
peak = np.where(llh_arr == max_llh)
#llh_arr = llh_arr / max_llh
#area2 = np.trapz(llh_arr, x=ss_arr)
#print(f'new area: {area2}')
'''

#same for dd plot
'''
dd_exp, energies, phi_xs = get_dd_function(2.0, is_experimental=True)
for i, spot_size in enumerate(ss_arr):
    dd_fun, _, _ = get_dd_function(spot_size)
    llh_arr[i] = get_2d_likelihood(energies, phi_xs, dd_exp, dd_fun)
peak = np.where(llh_arr == np.amax(llh_arr))

plt.plot(ss_arr, llh_arr)
plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
plt.xlabel('Beam $\sigma$ ($\mu m$)')
plt.ylabel('Likelihood')
plt.legend()
plt.savefig('mle2/dd_64_par_4.png', dpi = 400)
plt.clf()
'''

#dist llhs
'''
dist_exp, phi_xs, phi_ys, _ = get_dist_function(experimental_ss, is_experimental=True, name_extension='', num_part_name=100)
for i, spot_size in enumerate(ss_arr):
    dist_fun, _, _, dist = get_dist_function(spot_size, num_part_name=100)
    #llh_arr[i] = get_2d_likelihood_twist(phi_xs, phi_ys, dist_exp, dist)
    llh_arr[i] = get_2d_likelihood(phi_xs, phi_ys, dist_exp, dist_fun)
    print(spot_size, llh_arr[i])
'''

'''
#simple dist mle
dist_exp, phi_xs, phi_ys = get_dist_function(2, is_experimental=True, name_extension='2')
ss_arr = np.array([1, 2, 3, 4, 5])
llh_arr = np.empty(len(ss_arr))
for i, spot_size in enumerate(ss_arr):
    dist_fun, _, _ = get_dist_function(spot_size)
    llh_arr[i] = get_2d_likelihood(phi_xs, phi_ys, dist_exp, dist_fun)
peak = np.where(llh_arr == np.amax(llh_arr))

plt.plot(ss_arr, llh_arr)
plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
plt.xlabel('Beam $\sigma$ ($\mu m$)')
plt.ylabel('Likelihood')
plt.legend()
plt.savefig('mle2/dist_32_par.png', dpi = 400)
plt.clf()
'''

#simple dd mle
'''
dd_exp, energies, phi_xs = get_dd_function(2, is_experimental=True)
ss_arr = np.array([1, 2, 3, 4, 5])
llh_arr = np.empty(len(ss_arr))
for i, spot_size in enumerate(ss_arr):
    dd_fun, _, _ = get_dd_function(spot_size)
    llh_arr[i] = get_2d_likelihood(energies, phi_xs, dd_exp, dd_fun)
peak = np.where(llh_arr == np.amax(llh_arr))

plt.plot(ss_arr, llh_arr)
plt.scatter(ss_arr[peak], llh_arr[peak], marker = 'x', color = 'red', label = 'maximum', s = 60.)
plt.xlabel('Beam $\sigma$ ($\mu m$)')
plt.ylabel('Likelihood')
plt.legend()
plt.savefig('mle2/dd_32_par2.png', dpi = 400)
plt.clf()
'''
