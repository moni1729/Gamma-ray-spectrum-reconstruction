#include "merit.hxx"
#include "radiation.hxx"
#include <cassert>
#include <cmath>
#include <memory>

double compute_merit(EnergyPhixPhiy* samples, std::size_t n_samples,
  std::size_t particles, std::size_t steps, unsigned seed, double spot_size,
  double normalized_emittance, double gamma_initial, double ion_atomic_number,
  double plasma_density, double accelerating_field, double time_step)
{
  auto radiation = std::make_unique<double[]>(n_samples * 6);
  get_beam_radiation(radiation.get(), samples, n_samples, particles, steps, seed,
    spot_size, normalized_emittance, gamma_initial, ion_atomic_number,
     plasma_density, accelerating_field, time_step);
  double sum = 0.0;
  for (std::size_t i = 0; i != n_samples; ++i) {
   sum += (radiation[6 * i + 0] * radiation[6 * i + 0]
     + radiation[6 * i + 1] * radiation[6 * i + 1]
     + radiation[6 * i + 2] * radiation[6 * i + 2]
     + radiation[6 * i + 3] * radiation[6 * i + 3]
     + radiation[6 * i + 4] * radiation[6 * i + 4]
     + radiation[6 * i + 5] * radiation[6 * i + 5]);
  };
  double merit = 0.0;
  for (std::size_t i = 0; i != n_samples; ++i) {
    double sample_merit = (radiation[6 * i + 0] * radiation[6 * i + 0]
      + radiation[6 * i + 1] * radiation[6 * i + 1]
      + radiation[6 * i + 2] * radiation[6 * i + 2]
      + radiation[6 * i + 3] * radiation[6 * i + 3]
      + radiation[6 * i + 4] * radiation[6 * i + 4]
      + radiation[6 * i + 5] * radiation[6 * i + 5]);
    merit += std::log(sample_merit / sum);
  };
  return merit;
}
