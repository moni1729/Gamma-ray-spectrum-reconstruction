import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.special
import scipy.integrate
import os
from scipy import signal


def linspace_midpoint(min, max, num):
    vals, step = np.linspace(min, max, num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((min - 0.5 * step), (max + 0.5 * step), (num + 1), endpoint=True)
    return (vals, vals_midpoint, step)

def logspace_midpoint(min, max, num):
    vals, step = np.linspace(np.log(min), np.log(max), num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((np.log(min) - 0.5 * step), (np.log(max) + 0.5 * step), (num + 1), endpoint=True)
    return np.exp(vals), np.exp(vals_midpoint), None


def plot_from_files(output_filename, rad_filename, plot_type='both'):


    output_dict = {}
    with open(output_filename, 'r') as f:
        for line in f:
            split = line.split()
            output_dict[split[0]] = (split[2], split[3])

    energies, energies_midpoint, _ = linspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
    phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
    phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
    particles = int(output_dict['actual_particles'][0])


    result_wit = np.fromfile(rad_filename).reshape((len(energies), len(phi_xs), len(phi_ys), 6))
    dd_wit = np.sum(result_wit ** 2, axis=3)
    dd_wit *= (0.5e-9 / (1.602176634e-19 * particles))

    #plot double differential
    if plot_type == 'dd' or plot_type == 'both':
        fig, ax = plt.subplots()
        dd2_wit = np.sum(dd_wit, axis=2) * phi_ys_step
        vmin, vmax = dd2_wit.min(), dd2_wit.max()
        hm = ax.pcolormesh(energies_midpoint, phi_xs_midpoint * 1e6, dd2_wit.T) #T is transpose
        ax.set_xlim(energies.min(), energies.max())
        ax.set_ylim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
        ax.set_xlabel('photon energy (eV)')
        ax.set_ylabel(f'$\\phi_x$ ($\\mu$rad)')
        cbar = fig.colorbar(hm, ax=ax)
        cbar.set_label('$\\frac{dI}{d\\phi_x}$ (eV)')

        fig.savefig(f'results7/dd_poster.png', dpi=300)
        plt.close(fig)


    #plot distribution (angular)
    if plot_type == 'dist':
        dist = np.sum(0.5 * (dd_wit[1:, :, :] + dd_wit[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
        vmin, vmax = dist.min(), dist.max()
        fig, ax = plt.subplots()
        #ax.set_title()
        cmap = 'viridis'
        ax.contour(phi_xs_midpoint[:-1] * 1e6, phi_ys_midpoint[:-1] * 1e6, dist.T,50, cmap=cmap,vmin=vmin, vmax=vmax)
        ax.set_xlim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
        ax.set_ylim(phi_ys.min() * 1e6, phi_ys.max() * 1e6)
        ax.set_xlabel('$\\phi_x$ ($\\mu$rad)')
        ax.set_ylabel('$\\phi_y$ ($\\mu$rad)')
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)
        cbar.set_label('$\\frac{dI}{d\\Omega}$ (eV)')

        fig.savefig(f'results7/ang_poster.png', dpi=300)
        plt.close(fig)


    #plot spectrum for witness
    #currently plots probability
    if plot_type == 'spec' or plot_type == 'both':
        fig, ax = plt.subplots()
        spectrum_wit = np.sum(dd_wit, axis=(1,2)) * phi_xs_step * phi_ys_step
        prob_spec = spectrum_wit / spectrum_wit.sum()
        #prob_spec = spectrum_wit / np.trapz(spectrum_wit, x=energies)
        #print(f'final area: {np.trapz(prob_spec, x=energies)}')
        ax.plot(energies, spectrum_wit, label='raw spectrum')#, color='black')
        ax.set_xlabel('photon energy (eV)')
        #ax.set_ylabel('probability')
        #ax.legend()
        #ax.set_ylabel(f'$f (x | \sigma$ = {spot_size})')
        ax.set_ylabel('$\\frac{dI}{d\\epsilon}$')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        fig.savefig(f'results7/spec_poster.png', dpi=300)
        plt.close(fig)


#main


num_particles = 64
folder_number = 4
plot_type = 'spec'
parameters = np.array(['sigma_x, sigma_y'])
parameter = 'ss'
param_units = 'micron' #for file names
unit_multiplier = 1e-6

plot_from_files('output', 'radiation', plot_type='spec')
plot_from_files('output', 'radiation', plot_type='dd')
plot_from_files('output', 'radiation', plot_type='dist')
