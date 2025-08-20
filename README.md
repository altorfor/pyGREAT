# pyGREAT

pyGREAT is a Python wrapper for the GREAT (General Relativistic Eigenmode Analysis Tool) Fortran code.  
It runs GREAT in-memory and writes structured HDF5 outputs:

- `background.h5` – background quantities (one group per time step)
- `eigen.h5` – eigenfunctions per time step / per mode
- `freqs.h5` – eigenfrequencies per time step

## Layout

fortran/ # shim + Makefile → builds libpygreat.{dylib,so}
pygreat/ # Python package (core + I/O)
scripts/ # CLI runner
parameters # GREAT-style parameter file

shell
Copy
Edit

## Build

cd fortran
make clean && make # produces fortran/lib/libpygreat.{dylib|so}

shell
Copy
Edit

## Run

python scripts/run_from_parfile.py --parfile parameters

nginx
Copy
Edit

