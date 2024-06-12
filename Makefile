all: sph-solver.cu
	nvcc -gencode arch=compute_75,code=sm_75 -o sph-solver sph-solver.cu 
clean:
	rm sph-solver
