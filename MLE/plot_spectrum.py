import ctypes
import numpy as np
import matplotlib.pyplot as plt

lib = ctypes.cdll.LoadLibrary('libbetatrondiagnostics.so')


def get_beam_trajectories(particles, step, steps, seed, spot, emit, gev,
    Z, n0, gradient):
    particle_data = np.empty(dtype=np.float64, shape=(particles, steps + 1, 9))
    lib.get_beam_trajectories(
        particle_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(particles),
        ctypes.c_size_t(steps),
        ctypes.c_size_t(seed),
        ctypes.c_double(spot),
        ctypes.c_double(emit),
        ctypes.c_double(1000 * gev / 0.510998),
        ctypes.c_double(Z),
        ctypes.c_double(n0),
        ctypes.c_double(gradient),
        ctypes.c_double(step)
    );
    return particle_data

def get_beam_radiation(inputs, particles, step, steps, seed, spot, emit, gev,
    Z, n0, gradient):
    n_inputs = inputs.shape[0]
    result = np.empty(dtype=np.float64, shape=(n_inputs, 6))
    lib.get_beam_radiation(
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_inputs),
        ctypes.c_size_t(particles),
        ctypes.c_size_t(steps),
        ctypes.c_size_t(seed),
        ctypes.c_double(spot),
        ctypes.c_double(emit),
        ctypes.c_double(1000 * gev / 0.510998),
        ctypes.c_double(Z),
        ctypes.c_double(n0),
        ctypes.c_double(gradient),
        ctypes.c_double(step)
    );
    return result

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

c_light = 299792458

def trajectory(t, trajectories, filename):
    assert trajectories.shape[0] == 1
    x = trajectories[0, :, 0]
    y = trajectories[0, :, 1]
    bx = trajectories[0, :, 3]
    by = trajectories[0, :, 4]
    r = np.sqrt(x ** 2 + y ** 2)
    br = (x * bx + y * by) / r
    z_cm = 100 * (trajectories[0, :, 2] + c_light * t)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.plot(z_cm, x, label='x (m)')
    ax1.plot(z_cm, y, label='y (m)')
    ax1.plot(z_cm, r, label='r (m)')
    ax1.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
    ax1.legend()
    ax2.plot(z_cm, bx, label='$\\beta_x$')
    ax2.plot(z_cm, by, label='$\\beta_y$')
    ax2.plot(z_cm, br, label='$\\beta_r$')
    ax2.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
    ax2.legend()
    ax2.set_xlabel('z (cm)')
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def phase_space(trajectories, filename):
    assert trajectories.shape[0] == 1
    x = trajectories[0, :, 0]
    y = trajectories[0, :, 1]
    bx = trajectories[0, :, 3]
    by = trajectories[0, :, 4]
    g = trajectories[0, :, 5]
    r = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(y, x)
    br = (x * bx + y * by) / r
    bth = (x * by - y * bx) / (x ** 2 + y ** 2)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(plt.rcParams['figure.figsize'][0] * 2, plt.rcParams['figure.figsize'][1] * 1))
    ax1.plot(x, g * bx)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('$\\gamma \\beta_x$')
    ax1.get_xaxis().get_major_formatter().set_powerlimits((0, 2))
    ax1.get_yaxis().get_major_formatter().set_powerlimits((0, 2))
    ax2.plot(y, g * by)
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('$\\gamma \\beta_y$')
    ax2.get_xaxis().get_major_formatter().set_powerlimits((0, 2))
    ax2.get_yaxis().get_major_formatter().set_powerlimits((0, 2))
    ax3.plot(r, g * br)
    ax3.set_xlabel('r (m)')
    ax3.set_ylabel('$\\gamma \\beta_r$')
    ax3.get_xaxis().get_major_formatter().set_powerlimits((0, 2))
    ax3.get_yaxis().get_major_formatter().set_powerlimits((0, 2))
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def transverse(trajectories, filename):
    assert trajectories.shape[0] == 1
    fig, ax = plt.subplots()
    limit = 1.05 * np.sqrt(trajectories[0, :, 0] ** 2 + trajectories[0, :, 1] ** 2).max()
    ax.plot(trajectories[0, :, 0], trajectories[0, :, 1])
    ax.plot(trajectories[0, 0, 0], trajectories[0, 0, 1], 'ro')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.get_xaxis().get_major_formatter().set_powerlimits((0, 1))
    ax.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
    fig.savefig(filename, dpi=300)
    plt.close(fig)


part = 100
dist = 0.01
steps = 2000
step = dist / (299792458 * steps)

r'''
trajectory(t, res, 'foo.png')
phase_space(res, 'foo2.png')
transverse(res, 'foo3.png')

plt.plot(t, res[0,:,2])
plt.savefig('2.png')
plt.clf()

plt.plot(t, res[0,:,5])
plt.savefig('5.png')
plt.clf()

'''

ens = np.linspace(1e3, 1e7, 100)
inputs = np.array([[en, 0.0, 0.0] for en in ens])
res = get_beam_radiation(inputs, part, step, steps,
    1000, 3.4e-6, 3.4e-6, 0.75, 1, 1e17 * 100 * 100 * 100, 0)

plt.plot(ens, np.sum(res ** 2, axis=1))
plt.savefig('bar2.png')
plt.clf()

ens = np.logspace(3, 7, 100)
inputs = np.array([[en, 0.0, 0.0] for en in ens])
res = get_beam_radiation(inputs, part, step, steps,
    4329, 3.4e-6, 3.4e-6, 0.75, 1, 1e17 * 100 * 100 * 100, 0)

plt.loglog(ens, np.sum(res ** 2, axis=1))
plt.savefig('barlog2.png')
plt.clf()
