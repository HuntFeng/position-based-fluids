#include "particle.cu"
#include "utils.cu"
#include "vector.cu"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define h 0.5       // smooth radius
#define N 500       // number of particles
#define L 1.0       // system size
#define steps 500   // total steps
#define diag_step 5 // diagnostics step

// return random number between [0.01,1)
double rng() { return 0.01 + 0.98 * ((double)std::rand() / RAND_MAX); }

void init_particles(thrust::host_vector<Particle> &particles) {
  for (Particle &p : particles) {
    p.r = Vector<double>({rng() * L, rng() * L});
    p.v = Vector<double>({0, 0});
    p.m = 1.0;
    p.rho = 0;
    p.p = 0;
  }
}

// __device__ double w(Vector r) {
//   return 1 / pow(sqrt(M_PI) * h, DIM) * exp(-pow(r.norm(), 2) / pow(h, 2));
// }
//
// __device__ Vector grad_w(Vector r) { return -(2 / pow(h, 2)) * r * w(r); }

__device__ double w(Vector<double> r) {
  double r_norm = r.norm();
  if (r_norm >= h)
    return 0;
  else {
    double normalized_const;
    if (DIM == 1)
      normalized_const = 3 / pow(h, 3);
    else if (DIM == 2)
      normalized_const = 6 / (M_PI * pow(h, 4));
    else
      normalized_const = 15 / (2 * M_PI * pow(h, 5));
    return normalized_const * pow(h - r_norm, 2);
  }
}

__device__ Vector<double> grad_w(Vector<double> r) {
  double r_norm = r.norm();
  if (r_norm >= h)
    return Vector<double>({0, 0});
  else if (r_norm == 0)
    return Vector<double>({0, 0});
  else {
    double normalized_const;
    if (DIM == 1)
      normalized_const = 3 / pow(h, 3);
    else if (DIM == 2)
      normalized_const = 6 / (M_PI * pow(h, 4));
    else
      normalized_const = 15 / (2 * M_PI * pow(h, 5));

    return normalized_const * -2 * (h / r_norm - 1) * r;
  }
}

__global__ void calculate_logical_position(Particle *p) {
  int i = threadIdx.x;
  for (int d = 0; d < DIM; d++)
    p[i].l[d] = int(p[i].r[d] / h);
}

__global__ void calculate_density(Particle *p) {
  int i = threadIdx.x;
  for (int j = 0; j < N; j++)
    p[i].rho += p[j].m * w(p[i].r - p[j].r);
}

__global__ void calculate_pressure(Particle *p) {
  int i = threadIdx.x;
  // double rho0 = N * p[0].m / L * L;
  // double k = 0.5;
  // p[i].p = k * pow(p[i].rho - rho0, 2);
  double k = 100;
  p[i].p = k * p[i].rho;
}

__global__ void calculate_acceleration(Particle *p) {
  int i = threadIdx.x;
  p[i].a = Vector<double>({0, -10});
  for (int j = 0; j < N; j++)
    p[i].a -= p[j].m * (p[i].p / pow(p[i].rho, 2) + p[j].p / pow(p[j].rho, 2)) *
              grad_w(p[i].r - p[j].r);
}

__global__ void push_particles(Particle *p) {
  double dt = 0.01;
  // update velocity and position using leap-frog
  int i = threadIdx.x;

  p[i].v += p[i].a * dt;
  p[i].r += p[i].v * dt;
}

__global__ void apply_boundary(Particle *p) {
  int i = threadIdx.x;

  for (int d = 0; d < DIM; d++) {
    if (p[i].r[d] <= 0) {
      p[i].r[d] = -p[i].r[d];
      p[i].v[d] = -0.5 * p[i].v[d];
    }

    if (p[i].r[d] >= L) {
      p[i].r[d] = 2 - p[i].r[d];
      p[i].v[d] = -0.5 * p[i].v[d];
    }
  }
}

int main() {
  thrust::host_vector<Particle> h_particles(N);
  init_particles(h_particles);

  thrust::device_vector<Particle> d_particles = h_particles;
  Particle *particles = thrust::raw_pointer_cast(d_particles.data());

  clock_t begin = clock();

  calculate_density<<<1, N>>>(particles);
  calculate_pressure<<<1, N>>>(particles);
  calculate_acceleration<<<1, N>>>(particles);
  thrust::copy(d_particles.begin(), d_particles.end(), h_particles.begin());
  prepare_data_dir();
  save_particles(h_particles, 0);

  for (int step = 1; step <= steps; step++) {
    calculate_density<<<1, N>>>(particles);
    calculate_pressure<<<1, N>>>(particles);
    calculate_acceleration<<<1, N>>>(particles);
    push_particles<<<1, N>>>(particles);
    apply_boundary<<<1, N>>>(particles);

    printf("step = %d \n", step);
    if (step % diag_step == 0) {
      thrust::copy(d_particles.begin(), d_particles.end(), h_particles.begin());
      save_particles(h_particles, step / diag_step);
    }
  }

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("took %f seconds\n", time_spent);
  return 0;
}
