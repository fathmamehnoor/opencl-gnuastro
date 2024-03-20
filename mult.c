#include <stdio.h>
#include <CL/cl.h>

char * get_kernel_source(char *filename)
{
  int size;
  FILE *fp = fopen(filename, "r");

  if(!fp)
    {
      fprintf(stderr, "Error: Could not open kernel source file\n");
      exit(EXIT_FAILURE);
    }
  fseek(fp, 0, SEEK_END);
  size = ftell(fp) + 1;
  rewind(fp);

  char *source = (char *)calloc(sizeof(char), size);
  if(!source)
    {
      fprintf(stderr, "Error: Could not allocate memory for kernel" 
             "source string\n");
      exit(EXIT_FAILURE);
    }
  fread(source, sizeof(char), size, fp);
  fclose(fp);
  return source;
}


int main()
{
           
  int N;
  FILE *fp;              
  int size;
  size_t len;
  cl_int err;
  cl_kernel kernel;
  char buffer[2048];
  cl_program program;    
  cl_context context;
  char * kernelsource; 
  cl_device_id device;
  cl_mem d_A, d_B, d_C;
  float *h_A, *h_B, *h_C; 
  cl_platform_id platform;
  cl_command_queue commands; 

  const unsigned int block_size = 16;   
      

  N = 10000;

  size = N * N;

  /* Allocate host memory for matrices */
  h_A = (float *)malloc(size * sizeof(float));
  h_B = (float *)malloc(size * sizeof(float));
  h_C = (float *)malloc(size * sizeof(float));

  /* Read matrix data */
  fp = fopen("test1.fits", "rb");
  if(!fp) 
    {
      perror("Error opening file");
      return 1;
    }
  fread(h_A, sizeof(float), size, fp);

  fp = fopen("test2.fits", "rb");
  if(!fp)
    {
      perror("Error opening file");
      return 1;
    }
  fread(h_B, sizeof(float), size, fp);

  
  /* Initialize OpenCL */
  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  commands = clCreateCommandQueueWithProperties(context, device, 0, &err);
  

  /* Create OpenCL buffers for matrices */
  d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, h_A, &err);
  
  d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, h_B, &err);

  d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                          sizeof(float) * size, NULL, &err);

  /*  Read OpenCL kernel source code from file */
  kernelsource = get_kernel_source("kernel.cl");

  /* Create OpenCL program from source code */
  program = clCreateProgramWithSource(context, 1, (const char **) 
                                      & kernelsource, NULL, &err);
  free(kernelsource);
  
  /* Build OpenCL program */
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err != CL_SUCCESS)
    {
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                            sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
    }

  /* Create OpenCL kernel */
  kernel = clCreateKernel(program, "mult", &err);
  

  printf("Matrix multiplication, order %d on device\n",N);

  /* Set kernel arguments */
  err =  clSetKernelArg(kernel, 0, sizeof(int),    &N);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_A);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_B);
  err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_C);
  err |= clSetKernelArg(kernel, 4, sizeof(float) * block_size * block_size,
                        NULL);
  err |= clSetKernelArg(kernel, 5, sizeof(float) * block_size * block_size,
                        NULL);

  /* Define global and local work sizes for kernel execution */   
  const size_t global[2] = {N, N};
  const size_t local[2] = {block_size, block_size};

  /* Enqueue kernel for execution*/
  err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0,
                                NULL, NULL);
  
  /* Wait for kernel execution to finish */
  err = clFinish(commands);

  /* Read result back from device */
  err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, sizeof(float) * size, h_C,
                            0, NULL, NULL);

  /* cleanup */
  free(h_A);
  free(h_B);
  free(h_C);
  fclose(fp);
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}


