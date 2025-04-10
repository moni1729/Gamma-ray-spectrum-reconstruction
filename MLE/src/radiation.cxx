#include "radiation.hxx"
#include "trajectory.hxx"
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>

static constexpr double c_light = 299792458.0;
static constexpr double hbar_ev = 6.582119569e-16; // eV * s
static constexpr double constant = 0.013595738304; // sqrt(e^2 / (16 pi^3 epsilon_0 hbar c))

static std::array<double, 3> cross(std::array<double, 3> a, std::array<double, 3> b)
{
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  };
}

static std::array<double, 3> subtract(std::array<double, 3> a, std::array<double, 3> b)
{
  return {
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2]
  };
}

static double dot(std::array<double, 3> a, std::array<double, 3> b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void compute_radiation(double* __restrict__ result,
  Particle* __restrict__ data, EnergyPhixPhiy* __restrict__ inputs,
  std::size_t n_particles, std::size_t n_steps, std::size_t n_inputs,
  double time_step)
{
  for (std::size_t j = 0; j != n_inputs; ++j) {
    double frequency = inputs[j].energy / hbar_ev;
    double phi_x = inputs[j].phi_x;
    double phi_y = inputs[j].phi_y;

    std::size_t result_index = 6 * j;

    for (std::size_t a = 0; a != 6; ++a)
      result[result_index + a] = 0.0;

    for (std::size_t m = 0; m != n_particles; ++m) {
      for (std::size_t o = 0; o != n_steps + 1; ++o) {

        double t = time_step * o;

        Particle particle = data[o + m * n_steps];

        std::array<double, 3> n;
        std::array<double, 3> b;
        std::array<double, 3> bd;
        n[0] = std::sin(phi_x);
        n[1] = std::sin(phi_y);
        n[2] = std::sqrt(1 - std::pow(n[0], 2) - std::pow(n[1], 2));
        b[0] = particle.bx;
        b[1] = particle.by;
        b[2] = std::sqrt(1 - std::pow(particle.g, -2) - std::pow(particle.bx, 2) - std::pow(particle.by, 2));
        bd[0] = particle.bxd;
        bd[1] = particle.byd;
        bd[2] = -(particle.gd * std::pow(particle.g, -3) + particle.bx * particle.bxd + particle.by * particle.byd) / b[2];
        auto vector = cross(n, cross(subtract(n, b), bd));
        auto denom = std::pow(1 - dot(b, n), -2);
        double n_transverse2 = std::pow(n[0], 2) + std::pow(n[1], 2);
        double value = 0.5 * n_transverse2 + 0.125 * std::pow(n_transverse2, 2) + 0.0625 * std::pow(n_transverse2, 3);
        double phase = frequency * ((t * value) - (n[0] * particle.x + n[1] * particle.y + n[2] * particle.zeta) / c_light);
        double exponential_real = std::cos(phase);
        double exponential_imag = std::sin(phase);
        double integration_multiplier = (o == 0 || o == n_steps) ? 0.5 : 1.0;

        result[result_index + 0] += vector[0] * exponential_real * denom * integration_multiplier * time_step * constant;
        result[result_index + 1] += vector[0] * exponential_imag * denom * integration_multiplier * time_step * constant;
        result[result_index + 2] += vector[1] * exponential_real * denom * integration_multiplier * time_step * constant;
        result[result_index + 3] += vector[1] * exponential_imag * denom * integration_multiplier * time_step * constant;
        result[result_index + 4] += vector[2] * exponential_real * denom * integration_multiplier * time_step * constant;
        result[result_index + 5] += vector[2] * exponential_imag * denom * integration_multiplier * time_step * constant;
      }
    }
  }
}

void get_beam_radiation(double* radiation, EnergyPhixPhiy* samples,
  std::size_t n_samples, std::size_t particles, std::size_t steps,
  std::size_t seed, double spot_size, double normalized_emittance,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double accelerating_field, double time_step)
{
  auto particle_data = std::make_unique<Particle[]>(particles * (steps + 1));
  track_beam_blowout(particle_data.get(), particles, steps, seed, spot_size,
    normalized_emittance, gamma_initial, ion_atomic_number, plasma_density,
    accelerating_field, time_step);
  compute_radiation(radiation, particle_data.get(), samples,
    particles, steps, n_samples, time_step);
}
