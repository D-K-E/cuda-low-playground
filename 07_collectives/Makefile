# Needs MPI: sudo apt install openmpi-bin libopenmpi-dev
allreduce: allreduce.cpp
	mpicxx -O2 -std=c++17 -Wall -Wextra $< -o $@

run: allreduce
	mpirun $(MPIRUN_FLAGS) -np 4 ./allreduce

clean:
	rm -f allreduce

.PHONY: run clean
