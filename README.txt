
DOUBLE DIVIDE AND CONQUER SINGULAR VALUE DECOMPOSITION ON THE GPU

Authors:  Travis Askham, Mike Lewis, Steven Delong


setup:
   Make sure either your C_INCLUDE_PATH or the variable OPENCL_INC is pointing to the directory where
   your openCL include files are  (This directory should have the CL subdirectory)
      
    /my/path/to/include

   likewise, make sure either the LD_LIBRARY_PATH or OPENCL_LIB variables have the path to your 
   opencl library files.
 
     /home/my/path/to/lib/x86_64

  Then change to the ddc-svd directory and type make.  You should have the executable 
  test-whole-svd.  This can be run with
   
   ./test-whole-svd m n

   where m and n are integers for the number of rows and columns of a random matrix that will
   be decomposed into it's singular values and vectors.



   To use the svd in your own code, you need to use the svd_gpu function.  You can find the details of this
   function in svd_gpu.c.  

   To compile and run it, follow the example of test-whole-svd.c in terms of the includes, the Makefile, 
   and the function call itself.
