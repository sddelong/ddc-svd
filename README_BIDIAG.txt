Bidiagonalization Routines

Authors: Travis Askham (primary), Steven Delong
Email: taskham@nyu.edu

The bidiagonalization step of the provided SVD algorithm can be run on its own.
To see its usage, consult the driver file bidiag_dr.c. The program bidiag_dr 
can be compiled via:

	make bidiag_dr

and run via:

	./bidiag_dr m n

where m and n determine the dimensions of the test matrix built by the program. As
with the main SVD routine, you must have an appropriate OpenCL implementation and 
environment variables (see README.txt).  

The details of the bidiagonalization routines can be found in the .c files associated
with each library. There is a parallel library in

	bidiag_par.c

with header file

	bidiag_par.h

which includes an OpenCL implementation of the Golub-Kahan bidiagonalization routine.
This routine depends on all of the supplied .cl files. We have also included a
serial bidiagonalization routine which was written to help prototype and which is 
suitable for small matrix sizes. The details of this library can be found in 

	bidiag.c

with header file

	bidiag.h

All of these libraries depend on the routines found in matrix_helper.c and
cl-helper.c
