#ifndef VECTOR_CU
#define VECTOR_CU

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#define DIM 2

template <class T> class Vector {
public:
  T data[DIM];

  __host__ __device__ Vector() {
    for (int i = 0; i < DIM; i++)
      data[i] = 0;
  }

  __host__ __device__ Vector(std::initializer_list<T> components) {
    int i = 0;
    for (T val : components) {
      data[i] = val;
      i++;
    }
  }

  __host__ __device__ T &operator[](size_t index) { return data[index]; }

  __host__ __device__ const T &operator[](size_t index) const {
    return data[index];
  }

  __host__ __device__ Vector operator+(Vector const &other) {
    Vector result;
    for (int d = 0; d < DIM; d++)
      result[d] = data[d] + other[d];
    return result;
  }

  __host__ __device__ Vector operator+=(Vector const &other) {
    for (int d = 0; d < DIM; d++)
      data[d] += other[d];
    return *this;
  }

  __host__ __device__ Vector operator-(Vector const &other) {
    Vector result;
    for (int d = 0; d < DIM; d++)
      result[d] = data[d] - other[d];
    return result;
  }

  __host__ __device__ Vector operator-=(Vector const &other) {
    for (int d = 0; d < DIM; d++)
      data[d] -= other[d];
    return *this;
  }

  // for vector * scalar multiplication
  __host__ __device__ Vector operator*(double scalar) {
    Vector result;
    for (int d = 0; d < DIM; d++)
      result[d] = data[d] * scalar;
    return result;
  }

  __host__ __device__ Vector operator/(double scalar) {
    Vector result;
    for (int d = 0; d < DIM; d++)
      result[d] = data[d] / scalar;
    return result;
  }

  __host__ __device__ T norm() {
    T result = 0;
    for (int d = 0; d < DIM; d++)
      result += pow(data[d], 2);
    return sqrt(result);
  }

  __host__ __device__ void print() {
    printf("(");
    for (int d = 0; d < DIM - 1; d++)
      printf("%f, ", data[d]);
    printf("%f)\n", data[DIM - 1]);
  }
};

// for scalar * vector
__host__ __device__ Vector<double> operator*(double scalar,
                                             const Vector<double> &vec) {
  Vector<double> result;
  for (int d = 0; d < DIM; d++)
    result[d] = vec[d] * scalar;
  return result;
}

#endif
