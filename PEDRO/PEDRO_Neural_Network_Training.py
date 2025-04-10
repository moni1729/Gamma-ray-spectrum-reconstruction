#Importing relevant libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
from scipy.interpolate import interp1d
from scipy.integrate import simps
import scipy.stats


#Declaring constants
detector_bin_count = 128
twopi = 2*(np.pi)
MeV = 1.602176565e-13
GeV = 1.602176565e-10
detection_factor = 1.0
model_correction_factor = 8.1e-07

#Update this code_directory path depending on where this notebook and folder are located
code_directory_path = "/home/PEDRO/"

#Declaring relevant functions

#Implementation of the MLE algorithm
def iterate_shepp_vardi(x0, R, y):
  y0 = np.matmul(x0, R)
  mask = (y0 != 0)
  yrat = np.zeros_like(y)
  yrat[mask] = y[mask]/y0[mask]
  return (x0/R.sum(axis=1)) * np.matmul(R, yrat)

def load_response(fin):
  num_events = fin['num_events'][:].flatten()
  num_gamma_bins = len(num_events)
  gamma_bins = fin['i0'][:]*MeV
  hits = fin['hits'][:]
  detector_bins = fin['detector_bin'][:]
  photon_bins = fin['photon_bin'][:]*MeV

  # sum over photon energies
  hits = hits.sum(axis=2) * detection_factor

  hits = hits.astype('float64')
  num_cells = hits.shape[1]

  R = hits/num_events[:,np.newaxis]
  return R, gamma_bins, detector_bins

def photon_spectral_density_func(E):
  E0 = 1*GeV
  sigma0 = 2*GeV
  total_num_photons = 1e10
  A0 = total_num_photons/np.sqrt(twopi*sigma0**2)
  return A0 * np.exp(-(E-E0)**2/(2*sigma0**2))

#Plotting spectral photon density for the Nonlinear Compton Scattering case
def plot_spectral_photon_density_ncs(
        ax, filename, E_lim, num_photons_simulated, gamma_bins):
    photon_energy, spectral_photon_density = np.loadtxt(filename).T
    photon_energy *= GeV
    spectral_photon_density *= (1/(0.05*GeV))

    photon_energy, spectral_photon_density = resample(
        photon_energy, spectral_photon_density, 1000)

    mask = spectral_photon_density>0

    idx = range(*photon_energy.searchsorted(E_lim))
    num_photons_lim = simps(spectral_photon_density[idx], photon_energy[idx])
    alpha = num_photons_simulated/num_photons_lim

    ax.loglog(
        np.concatenate(((0.0,), photon_energy[mask]/GeV)),
        alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))),
        linewidth=0.6, label='Incident gamma spectrum')
    return generate_short_true_spectrum(np.concatenate(((0.0,), photon_energy[mask]/GeV)),
        alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))))

    #photon_energy[mask][1:]/GeV * (alpha*(spectral_photon_density[mask][1:] - spectral_photon_density[mask][:-1])/(1/GeV))

def resample(x, y, N):
    # import ipdb
    # ipdb.set_trace()
    x_y = interp1d(x, y, kind='linear')
    x0 = x[0]
    x1 = x[-1]
    u = 10**np.linspace(np.log10(x0), np.log10(x1), N)
    u[0] = x0
    u[-1] = x1
    v = x_y(u)
    return u, v

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array-value)).argmin()
  return idx

def plot_spectral_photon_density(
        ax, fin, group_name, E_lim, num_photons_simulated, gamma_bins):
    mrad = 0.001
    joule = 1.0
    g = fin[group_name]
    photon_energy = g['energy'][:]*MeV
    thetax = g['thetax'][:]*mrad
    thetay = g['thetay'][:]*mrad
    d2W = g['d2W'][:]*joule/(mrad**2*MeV)
    dthetax = thetax[1]-thetax[0]
    dthetay = thetay[1]-thetay[0]

    spectral_energy_density = d2W.sum(axis=(1,2))*dthetax*dthetay
    spectral_photon_density = spectral_energy_density/photon_energy

    mask = spectral_energy_density>0

    idx = range(*photon_energy.searchsorted(E_lim))
    num_photons_lim = simps(spectral_photon_density[idx], photon_energy[idx])
    alpha = num_photons_simulated/num_photons_lim

    ax.loglog(
        np.concatenate(((0.0,), photon_energy[mask]/GeV)),
        alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))),
        linewidth=0.6, label='Incident gamma spectrum')
    return generate_short_true_spectrum(np.concatenate(((0.0,), photon_energy[mask]/GeV)),
        alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))))
    #photon_energy[mask]/GeV * (alpha * spectral_photon_density[mask]/(1/GeV))

def plot_spectrum(spec_to_plot, dE, gamma_bins, title):
  spectral_density = spec_to_plot/dE
  ax.loglog(
        gamma_bins/GeV,
        np.concatenate(((0.0,),spectral_density/(1/GeV))),
        linewidth=0.6, label=title, ds = "steps")

def generate_short_true_spectrum(real_x_values, real_y_values):
    with h5py.File('/content/drive/MyDrive/Colab Notebooks/PBPL/spectrum.h5', 'r') as fin:
        energy_bins = fin['energy_bins'][:]*MeV
        photon_spectral_density = fin['photon_spectral_density'][:]*(1/MeV)
        num_events = fin['num_events'][:]

    x_values = energy_bins[0:-1]/GeV
    y_values = photon_spectral_density/(1/GeV)

    short_gamma_distribution = []

    for i in range(0, len(x_values)):
        index = find_nearest(real_x_values, x_values[i])
        short_gamma_distribution.append(real_y_values[index])
    return short_gamma_distribution

# This function will attempt to reconstruct a given (or random) gamma distribution after converting into a PEDRO spectrum
def gamma_dist_pred_and_plot(energy_bin, dist = np.zeros(shape=(1,64))):
  arb_gamma_distrbution = np.zeros(shape=(1,64))

  if (np.array_equal(dist,arb_gamma_distrbution)):
    for i in range(0, len(energy_bin)):
      #Decreasing Sigmoid: 1e11*(1 - 1/(1+np.exp(-((i+1)/8 - 4)))), 1e11*(1 - 1/(1+np.exp(-((i+0)/8 - 4))))
      #Exponential: 1e10*np.exp(i/64), 1e10*np.exp((i+1)/64)
      #Logarithmic: (64-(i+1))*1e10/64, (64-i)*1e10/64
      #Random: 0, 1e10
      rand_freq = np.random.randint(0, 1e10)
      arb_gamma_distrbution[0][energy_bin[i]] = rand_freq
  else:
    arb_gamma_distrbution[0] = dist

  #Calculating the simulated spectrum by multiplying the distribution with the R matrix
  y_experiment = np.dot(arb_gamma_distrbution[0], R)

  #Applying the ML model to reconstruct the gamma distribution
  output=model(tf.convert_to_tensor([y_experiment]))
  num_events = output.numpy()[:]
  x=num_events[0]

  #Using the ML reconstruction as the initial guess for MLE
  x_s = x
  for i in range(75):
    x_s = iterate_shepp_vardi(x_s, R, y_experiment)

  scale_factor_list = []

  #Predicts the distribution 100 times and calculates the scale difference between the first element of the guess and original gamma distribution each time
  for i in range (0, 100):
    output=model(tf.convert_to_tensor([y_experiment]))
    num_events = output.numpy()[:]
    x=num_events[0]
    scale_factor_list.append(arb_gamma_distrbution[0][0]/x[0])

  #Calculating the average scale factor and printing it
  scale_factor = np.mean(scale_factor_list)
  print(scale_factor)
  scale_factor = 1

  #Plotting the ML guess (with the scale_factor correction) with the original gamma distribution
  gamma_dist_plot(x*scale_factor, arb_gamma_distrbution[0])

def gamma_dist_plot(prediction, true):
  qr_guess = recover_energy_dist(R, np.dot(prediction, R))
  plot.plot(prediction, label = "Model based prediction")
  #plot.plot(qr_guess[0], label = "QR based prediction")
  plot.plot(true, label = "Gamma distribution")
  plot.title("Comparison of Gamma Energy Distributions")
  plot.xlabel("Energy Bin")
  plot.ylabel("Number of Photons")
  plot.yscale("log")
  plot.legend()

#Implementation of the QR decomposition-based reconstruction method
def recover_energy_dist(R, spectrum): #R*guess = spectrum
  Q,S = np.linalg.qr(np.matrix.transpose(R), mode = "complete") #Q*S*guess = spectrum
  b = np.array([np.matrix.transpose(Q).dot(spectrum)], dtype="float") #S*guess = Q^t * spectrum = b
  guess = np.linalg.lstsq(S,b[0]) # || S*guess - b ||_2 = 0
  return guess


#Loading the data
f = h5py.File(code_directory_path + "R.h5","r")
data = f["hits"]

#Constructing the data structures to pass to the model
R = np.empty(shape = (64, detector_bin_count), dtype = object) #R matrix
train_hits = np.empty(shape = (64, detector_bin_count), dtype = object) #MODEL INPUT
train_spectra = np.zeros(shape = (64, 64), dtype = object) #MODEL OUTPUT

#Creating the arrays to contain the testing data.
test_hits = np.zeros(shape = (1, detector_bin_count), dtype = object)
test_spectra = np.zeros(shape = (1, 64), dtype = object)

energy_bins = f["num_events"][:]
noise_level = 1e04

for i in range(0, 64):
  train_spectra[i][i] = energy_bins[i]
  vector = []
  for k in range(0, detector_bin_count):
    vector.append(detection_factor*sum(data[i][k])/1e09)
  R[i] = vector
  train_hits[i] = np.dot(train_spectra[i][i], R[i])

num_of_training_cases = 100

#The following loop randomly generates num_of_training_cases by randomly creating photon distributions and calculating the
#associated electron-positron hits to provide to the model as training data.
for i in range(0, num_of_training_cases):
  arb_gamma_distrbution = np.zeros(shape=(1,64))
  #Creating an array of bins to generate a random number of photons
  #Consider generating random weights within the space of all betatron functions
  #Should be a 2D search space
  #This can yield a library of neural networks trained on different physical cases
  #Eg: energy_bin = [0, 24, 36, 52, 14, 9, 61] when i%12 = 7
  #energy_bin = np.random.randint(0, 64, 64)

  for k in range(0, 64):
    arb_gamma_distrbution[0][k] = np.random.randint(1e05, 1e10) #Adding a random number of photons between 0 and 1e10 in each bin based on energy_bin

  #Calculating the simulated hits in the PEDRO detector by multiplying the arbitrary gamma distrbution with the R matrix
  calculated_hits = np.zeros(shape=(1,detector_bin_count))
  calculated_hits[0] = np.dot(arb_gamma_distrbution[0], R)
  calculated_hits = calculated_hits + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data

  #Adding the simulated hits and gamma distribution to the training data
  train_hits  = np.concatenate((train_hits, calculated_hits))
  train_spectra = np.concatenate((train_spectra, arb_gamma_distrbution))

#The following loop is creating testing data for the model.
#Consider adding noise to the testing data, but not to the training data?
test_cases = int(num_of_training_cases/10)
for i in range(0, test_cases):
  arb_gamma_distrbution = np.zeros(shape=(1,64))

  for k in range (0, 64):
    upper_lim = np.random.randint(5, 10)
    arb_gamma_distrbution[0][k] = np.random.randint(1e05, 10e10)

  calculated_hits = np.zeros(shape=(1,detector_bin_count))
  calculated_hits[0] = np.dot(arb_gamma_distrbution[0], R)
  calculated_hits = calculated_hits + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data

  test_hits  = np.concatenate((test_hits, calculated_hits))
  test_spectra = np.concatenate((test_spectra, arb_gamma_distrbution))

#Converting the arrays into formats the model can process
train_hits = tf.convert_to_tensor(train_hits, dtype=float)
train_spectra = tf.convert_to_tensor(train_spectra, dtype=float)
test_hits = tf.convert_to_tensor(test_hits, dtype=float)
test_spectra = tf.convert_to_tensor(test_spectra, dtype=float)


#For this model, I want to provide a vector/list of 128 elements and associate that with a 64-bin spectrum.
#2 layers proved to be more effective than 1 layer or 3 layers
model = keras.Sequential(
  [
   layers.Dense(64, activation='linear', use_bias=True, input_shape=(128,)),
   layers.Dense(64, activation='linear', use_bias=True, input_shape=(64,))
   ]
)

#Learning rate of 0.005 was chosen after hyperparameter tuning
opt = keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,loss=tf.keras.losses.mse, metrics='accuracy')
model.fit(train_hits, train_spectra, epochs = 400, verbose=0)
print("Evaluate on test data")
results = model.evaluate(test_hits, test_spectra)
print("test loss, test acc:", results)


#For the .h5 file, use the following key:
# nlcs.h5 for Nonlinear Compton Scatter
# filamentation.h5 for Filamentation
# qed.h5 for Quantum Electrodynamics
with h5py.File(code_directory_path + 'qed.h5', 'r') as fin:
  y_experiment = fin['hits'][0,:]
  y_experiment = y_experiment.sum(axis=1)
  y_experiment = y_experiment.astype('float64')
  num_events_real = fin['num_events'][:]

with h5py.File(code_directory_path + "R.h5","r") as f:
  R_test, gamma_bins, detector_bins = load_response(f)

dE = gamma_bins[1:] - gamma_bins[:-1]
energy = gamma_bins[:-1] + 0.5*dE
x0 = photon_spectral_density_func(energy)

#The following set of lines (until y_experiment = np.dot(arb_gamma_distrbution[0], R)) can be used to implement different cases
#Uncomment the following line and all lines associated with each case to test it. Don't forget to comment the last 4 lines of
#this code block to avoid loading the experimental cases

arb_gamma_distrbution = np.zeros(shape=(1,64)) #Setting the gamma distribution equal to 0

#Monoenergetic Spectrum
#arb_gamma_distrbution[0][2] = 1e08 #Adding one arbitrary spike (the second index indicates the bin)

noise_level = 1e02

#Bienergetic Spectrum
arb_gamma_distrbution[0][2] = 1e08 #Adding one arbitrary spike (the second index indicates the bin)
#arb_gamma_distrbution[0][42] = 1e10  #Adding another arbitrary spike (the second index indicates the bin)

#Random Spectrum across all 64 bins
#for i in range(0, 64):
#  rand_freq = np.random.randint(0, 1e10)
#  arb_gamma_distrbution[0][i] = rand_freq

#Calculating the hits in each detector by multiplying the gamma distribution by the R matrix
#y_experiment = np.dot(arb_gamma_distrbution[0], R) + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data
y_experiment += np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data
gamma_dist = GeV * arb_gamma_distrbution[0]/dE

#Adding Gaussian noise
#Take peak spectrometer signal, /2^8, treat that a standard deviation for a GD with a mean of 0, add to spectrometer signal and make sure there are no negative values

#Reconstructing the spectrum using QR decomposition
qr_guess = recover_energy_dist(R, y_experiment)

#Converting the experimental values into a reconstructed spectrum using the ML model
output=model(tf.convert_to_tensor([y_experiment]))
num_events = output[:]

model_guess = num_events[0]

#Reconstructing the spectrum using Maximum-Likelihood Estimation with the ML model's guess provided as the initial guess
mle_guess = model_guess
for i in range(5):
  mle_guess = iterate_shepp_vardi(mle_guess, R, y_experiment)

photon_spectral_density = model_guess[0]/dE
E_lim = [energy_bins[0], energy_bins[-1]]


#Code to write the information into a spectrum h5 file
with h5py.File(code_directory_path + 'spectrum.h5', 'w') as fout:
  fout['num_events'] = num_events_real
  fout['energy_bins'] = gamma_bins/MeV
  fout['photon_spectral_density'] = photon_spectral_density/(1/MeV)

#Include short paragraph about how we could use this to reconstruct multishot cases

matplotlib.rc(
        'figure.subplot', right=0.97, top=0.96, bottom=0.15, left=0.13)
fig = plot.figure(figsize=(12.0, 7.0))
ax = fig.add_subplot(1, 1, 1)

with h5py.File(code_directory_path + 'spectrum.h5', 'r') as fin:
        energy_bins = fin['energy_bins'][:]*MeV
        photon_spectral_density = fin['photon_spectral_density'][:]*(1/MeV)
        num_events = fin['num_events'][:]

plot.xlabel('Gamma energy (GeV)', labelpad=0.0)
plot.ylabel(r'Photon density (1/Gev)', labelpad=0.0)
photon_spectral_density = num_events[0]/dE

#Insert variable into the plot_spectral_photon_density depending on whether you're running filamentation or qed respectively
filamentation_group = '/Filamentation/solid'
sfqed_group = '/SFQED/MPIK/LCS+LCFA_w2.4_xi7.2'

#If testing the arbitrary gamma distributions, comment out all of the following lines
with h5py.File(code_directory_path + 'd2W.h5', 'r') as fin:
  #If running Nonlinear Compton scattering case, comment thef following line and uncomment the last line
  #gamma_dist = plot_spectral_photon_density(ax, fin, sfqed_group, E_lim, num_events, gamma_bins)
  print("test")

#Uncomment only when running Nonlinear Compton scattering case; else comment out
#gamma_dist = plot_spectral_photon_density_ncs(ax, code_directory_path +'nonlinear-compton-scattering.dat', E_lim, num_events, gamma_bins)

plot_spectrum(np.abs(model_guess.numpy()), dE, gamma_bins, "model")
plot_spectrum(np.abs(mle_guess), dE, gamma_bins, "mle")
plot_spectrum(np.abs(qr_guess[0]), dE, gamma_bins, "qr")

#Setting the title and finalizing the visual appearance of the Reconstruction Plot
ax.set_title("Filamentation Reconstructed (using _____ approach) and Gamma Spectra")
ax.legend()
locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
ax.yaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.yaxis.set_minor_locator(locmin)

ax.set_xlim(gamma_bins[0]/GeV, gamma_bins[-1]/GeV)
locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
ax.xaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.xaxis.set_minor_locator(locmin)
