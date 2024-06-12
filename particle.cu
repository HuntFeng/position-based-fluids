#ifndef PARTICLE_CU
#define PARTICLE_CU

#include "vector.cu"

typedef struct {
  // position
  Vector<double> r;
  // logical position (for faster neighbor search)
  Vector<int> l;
  // velocity
  Vector<double> v;
  // acceleration
  Vector<double> a;
  // mass
  double m;
  // density
  double rho;
  // pressure
  double p;
} Particle;

#endif
