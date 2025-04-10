#ifndef TRAJECTORY_HXX
#define TRAJECTORY_HXX

#include <cstddef>

struct Particle {
  double x, y, zeta, bx, by, g, bxd, byd, gd;
};

extern "C" {

void track_particle_bennett(Particle* result, double x0, double y0, double vx0,
  double vy0, double gamma_initial, double ion_atomic_number,
  double plasma_density, double rho_ion, double accelerating_field,
  double bennett_radius_initial, double time_step, std::size_t steps);

void track_particle_blowout(Particle* result, double x0, double y0, double vx0,
  double vy0, double gamma_initial, double ion_atomic_number,
  double plasma_density, double accelerating_field, double time_step,
  std::size_t steps);

void track_beam_blowout(Particle* result, std::size_t particles, std::size_t steps,
  std::size_t seed, double spot_size, double normalized_emittance,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double accelerating_field, double time_step);

}

#endif
