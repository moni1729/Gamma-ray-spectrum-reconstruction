#include "trajectory.hxx"
#include <algorithm>
#include <array>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cmath>
#include <cstddef>
#include <functional>

static constexpr double elementary_charge = 1.60217662e-19;
static constexpr double vacuum_permittivity = 8.8541878128e-12;
static constexpr double electron_mass = 9.1093837015e-31;
static constexpr double c_light = 299792458.0;

template <unsigned long n>
static std::array<double, n> arr_add(std::array<double, n> const& a, std::array<double, n> const& b)
{
  std::array<double, n> c;
  for (std::size_t i = 0; i != n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <unsigned long n>
static std::array<double, n> arr_mul(double a, std::array<double, n> const& b)
{
  std::array<double, n> c;
  for (std::size_t i = 0; i != n; ++i) {
    c[i] = a * b[i];
  }
  return c;
}

template <unsigned long n>
static void rk4(std::array<double, n> y0, std::function<std::array<double, n>(double, std::array<double, n>)> f, std::function<void(double, std::array<double, n>)> write, std::size_t steps, double time_step)
{
  std::array<double, n> y = y0;
  write(0.0, y);
  for (std::size_t step = 0; step != steps; ++step) {
    double t = step * time_step;

    #ifdef EULER

    // euler method
    y = arr_add(y, arr_mul(time_step, f(t, y)));

    #else

    // 4th order runge-kutta
    std::array<double, n> k1 = arr_mul(time_step, f(t, y));
    std::array<double, n> k2 = arr_mul(time_step, f(t + 0.5 * time_step, arr_add(y, arr_mul(0.5, k1))));
    std::array<double, n> k3 = arr_mul(time_step, f(t + 0.5 * time_step, arr_add(y, arr_mul(0.5, k2))));
    std::array<double, n> k4 = arr_mul(time_step, f(t + time_step, arr_add(y, k3)));
    y = arr_add(y, arr_mul(1/6.0, arr_add(arr_add(k1, k4), arr_mul(2.0, arr_add(k2, k3)))));

    #endif

    write(t + time_step, y);
  }
}

void track_particle_bennett(Particle* result, double x0, double y0, double vx0,
  double vy0, double gamma_initial, double ion_atomic_number,
  double plasma_density, double rho_ion, double accelerating_field,
  double bennett_radius_initial, double time_step, std::size_t steps)
{
  std::array<double, 5> initial_coordinates;
  initial_coordinates[0] = x0;
  initial_coordinates[1] = y0;
  initial_coordinates[2] = 0;
  initial_coordinates[3] = gamma_initial * vx0 / c_light;
  initial_coordinates[4] = gamma_initial * vy0 / c_light;

  double a = (1 + std::pow(initial_coordinates[3], 2) + std::pow(initial_coordinates[4], 2)) * std::pow(gamma_initial, -2);

  double constant_1 = ion_atomic_number * std::pow(elementary_charge, 2) * plasma_density / (2 * vacuum_permittivity * electron_mass * c_light);
  double constant_2 = rho_ion / plasma_density;
  double constant_3 = std::pow(bennett_radius_initial, -2) * std::pow(gamma_initial, -0.5);
  double constant_4 = std::sqrt(std::pow(gamma_initial, 2) - std::pow(initial_coordinates[3], 2) - std::pow(initial_coordinates[4], 2) - 1);
  double constant_5 = -elementary_charge * accelerating_field / (electron_mass * c_light);
  double constant_6 = gamma_initial * (-0.5 * a - 0.125 * a * a - 0.0625 * a * a * a * a);

  std::function<std::array<double, 5>(double, std::array<double, 5>)> f = [constant_1, constant_2, constant_3, constant_4, constant_5, constant_6, gamma_initial](double t, std::array<double, 5> coordinates){
   double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_4 + constant_5 * t, 2));
   double value = -constant_1 * (1 + constant_2  / (1 + constant_3 * std::sqrt(gamma) * (std::pow(coordinates[0], 2) + std::pow(coordinates[1], 2))));
   std::array<double, 5> rhs;
   rhs[0] = coordinates[3] * c_light / gamma;
   rhs[1] = coordinates[4] * c_light / gamma;
   rhs[2] = c_light * (constant_6 + constant_5 * t + (gamma_initial - gamma)) / gamma;
   rhs[3] = value * coordinates[0];
   rhs[4] = value * coordinates[1];
   return rhs;
  };

  std::function<void(double, std::array<double, 5>)> write = [&result, constant_1, constant_2, constant_3, constant_4, constant_5](double t, std::array<double, 5> coordinates){
    double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_4 + constant_5 * t, 2));
    double bx = coordinates[3] / gamma;
    double by = coordinates[4] / gamma;
    double value = -constant_1 * (1 + constant_2  / (1 + constant_3 * std::sqrt(gamma) * (std::pow(coordinates[0], 2) + std::pow(coordinates[1], 2))));
    double gamma_dot = (constant_5 * (constant_4 + constant_5 * t) + value * (coordinates[0] * coordinates[3] + coordinates[1] * coordinates[4])) / gamma;
    double bxd = (value * coordinates[0] / gamma) - (coordinates[3] * gamma_dot * std::pow(gamma, -2));
    double byd = (value * coordinates[1] / gamma) - (coordinates[4] * gamma_dot * std::pow(gamma, -2));
    std::array<double, 9> new_coordinates{coordinates[0], coordinates[1], coordinates[2], bx, by, gamma, bxd, byd, gamma_dot};
    std::copy_n(new_coordinates.data(), 9, reinterpret_cast<double*>(result));
    ++result;
  };

  rk4(initial_coordinates, f, write, steps, time_step);
}

void track_particle_blowout(Particle* result, double x0, double y0, double vx0,
  double vy0, double gamma_initial, double ion_atomic_number,
  double plasma_density, double accelerating_field, double time_step,
  std::size_t steps)
{
  std::array<double, 5> initial_coordinates;
  initial_coordinates[0] = x0;
  initial_coordinates[1] = y0;
  initial_coordinates[2] = 0;
  initial_coordinates[3] = gamma_initial * vx0 / c_light;
  initial_coordinates[4] = gamma_initial * vy0 / c_light;

  double a = (1 + std::pow(initial_coordinates[3], 2) + std::pow(initial_coordinates[4], 2)) * std::pow(gamma_initial, -2);

  double constant_1 = ion_atomic_number * std::pow(elementary_charge, 2) * plasma_density / (2 * vacuum_permittivity * electron_mass * c_light);
  double constant_2 = std::sqrt(std::pow(gamma_initial, 2) - std::pow(initial_coordinates[3], 2) - std::pow(initial_coordinates[4], 2) - 1);
  double constant_3 = -elementary_charge * accelerating_field / (electron_mass * c_light);
  double constant_4 = gamma_initial * (-0.5 * a - 0.125 * a * a - 0.0625 * a * a * a * a);

  std::function<std::array<double, 5>(double, std::array<double, 5>)> f = [constant_1, constant_2, constant_3, constant_4, gamma_initial](double t, std::array<double, 5> coordinates){
   double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_2 + constant_3 * t, 2));
   std::array<double, 5> rhs;
   rhs[0] = coordinates[3] * c_light / gamma;
   rhs[1] = coordinates[4] * c_light / gamma;
   rhs[2] = c_light * (constant_4 + constant_3 * t + (gamma_initial - gamma)) / gamma;
   rhs[3] = -constant_1 * coordinates[0];
   rhs[4] = -constant_1 * coordinates[1];
   return rhs;
  };

  std::function<void(double, std::array<double, 5>)> write = [&result, constant_1, constant_2, constant_3](double t, std::array<double, 5> coordinates){
    double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_2 + constant_3 * t, 2));
    double bx = coordinates[3] / gamma;
    double by = coordinates[4] / gamma;
    double value = -constant_1;
    double gamma_dot = (constant_3 * (constant_2 + constant_3 * t) + value * (coordinates[0] * coordinates[3] + coordinates[1] * coordinates[4])) / gamma;
    double bxd = (value * coordinates[0] / gamma) - (coordinates[3] * gamma_dot * std::pow(gamma, -2));
    double byd = (value * coordinates[1] / gamma) - (coordinates[4] * gamma_dot * std::pow(gamma, -2));
    std::array<double, 9> new_coordinates{coordinates[0], coordinates[1], coordinates[2], bx, by, gamma, bxd, byd, gamma_dot};
    std::copy_n(new_coordinates.data(), 9, reinterpret_cast<double*>(result));
    ++result;
  };

  rk4(initial_coordinates, f, write, steps, time_step);
}

void track_beam_blowout(Particle* result, std::size_t particles,
  std::size_t steps, std::size_t seed, double spot_size,
  double normalized_emittance, double gamma_initial, double ion_atomic_number,
  double plasma_density, double accelerating_field, double time_step)
{
  double sigma_v = c_light * normalized_emittance / (gamma_initial * spot_size);
  boost::random::mt19937 rng{seed};
  boost::random::normal_distribution xy_dist{0.0, spot_size};
  boost::random::normal_distribution vxvy_dist{0.0, sigma_v};
  for (std::size_t particle = 0; particle != particles; ++particle) {
    double x0 = xy_dist(rng);
    double y0 = xy_dist(rng);
    double vx0 = vxvy_dist(rng);
    double vy0 = vxvy_dist(rng);
    track_particle_blowout(
      result + particle * (steps + 1), x0, y0, vx0, vy0,
      gamma_initial, ion_atomic_number, plasma_density, accelerating_field,
      time_step, steps
    );
  }
}
