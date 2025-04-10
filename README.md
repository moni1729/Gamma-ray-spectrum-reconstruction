# Reconstruction of Beam Parameters and Betatron Radiation Spectra Measured with a Compton spectrometer

This project investigates the reconstruction of beam parameters from betatron radiation spectra in plasma wakefield acceleration (PWFA) 
experiments using a combination of simulated data and experimental modeling. Central to the approach is a custom Python-based tracking code
that models radiation emission in the PWFA blowout regime by calculating spectra from Liénard-Wiechert fields. The analysis includes both
one-dimensional and double-differential representations of the radiation. To infer beam properties such as spot size, emittance, and energy, 
the project implements a simulation-driven Maximum Likelihood Estimation (MLE) framework alongside machine learning (ML) models based on
densely connected neural networks. These include spectral reconstruction using Expectation-Maximization (EM) algorithms and ML techniques
applied to both 1D and 2D radiation spectra, including image-based models that use angular distributions as input. A key innovation in 
this work is the application of spectral normalization techniques, particularly for improving prediction accuracy at small spot sizes
(low \( K_\sigma \)). The modeling and diagnostic tools are validated against Geant4 simulations and designed to align with realistic 
FACET-II constraints, enhancing their relevance for current and future experiments.

Machine learning-based analysis of experimental electron beams and gamma energy distributions(https://arxiv.org/abs/2209.12119)
