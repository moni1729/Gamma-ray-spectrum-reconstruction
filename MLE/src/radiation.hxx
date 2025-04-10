#ifndef RADIATION_HXX
#define RADIATION_HXX

#include <cstddef>

struct Particle;

struct EnergyPhixPhiy {
  double energy, phi_x, phi_y;
};

extern "C" {

void compute_radiation(double* __restrict__ result,
  Particle* __restrict__ data, EnergyPhixPhiy* __restrict__ inputs,
  std::size_t n_particles, std::size_t n_steps, std::size_t n_inputs,
  double time_step);

void get_beam_radiation(double* radiation, EnergyPhixPhiy* samples,
  std::size_t n_samples, std::size_t particles, std::size_t steps,
  std::size_t seed, double spot_size, double normalized_emittance,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double accelerating_field, double time_step);

}

#endif
