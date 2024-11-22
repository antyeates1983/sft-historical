# Set FORTRAN90 compiler
FC = gfortran

# Set compiler flags for optimized gfortran
FFLAGS = -O3 -Wall -Wextra -Wno-maybe-uninitialized -Wno-compare-reals -ffast-math -funroll-loops --param max-unroll-times=5 #-fopenmp #-fbounds-check

all: sft1d

sft1d: sft1d.f90
	$(FC) $(FFLAGS) -o sft1d sft1d.f90

clean:
	@rm -f *.mod *.o sft1d