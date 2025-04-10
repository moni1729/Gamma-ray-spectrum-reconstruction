#ifndef MERIT_HXX
#define MERIT_HXX

#include <cstddef>

struct EnergyPhixPhiy;

extern "C" {

double compute_merit(EnergyPhixPhiy* samples, std::size_t n_samples,
  std::size_t particles, std::size_t steps, unsigned seed, double spot_size,
  double normalized_emittance, double gamma_initial, double ion_atomic_number,
  double plasma_density, double accelerating_field, double time_step);

}

#endif
