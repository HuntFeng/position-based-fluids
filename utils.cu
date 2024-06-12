#include "particle.cu"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thrust/host_vector.h>

// void print_particles(std::vector<Particle> particles) {
void print_particles(thrust::host_vector<Particle> particles) {
  for (Particle particle : particles) {
    for (int d = 0; d < DIM; d++) {
      printf("r[%d]=%f ", d, particle.r[d]);
    }
    printf("\n");
  }
}

void prepare_data_dir() {
  bool is_data_dir_exist = std::filesystem::exists("data");
  if (is_data_dir_exist)
    std::filesystem::remove_all("data");
  std::filesystem::create_directory("data");
}

// r, v, a, rho, p, m
void save_particles(thrust::host_vector<Particle> particles, int frame) {
  std::string name = "data/" + std::to_string(frame) + ".csv";
  std::ofstream file(name);
  for (Particle particle : particles) {
    for (int d = 0; d < DIM; d++) {
      file << particle.r[d] << ", ";
    }
    for (int d = 0; d < DIM; d++) {
      file << particle.v[d] << ", ";
    }
    for (int d = 0; d < DIM; d++) {
      file << particle.a[d] << ", ";
    }
    file << particle.rho << ", ";
    file << particle.p << ", ";
    file << particle.m << "\n";
  }
  file.close();
}
