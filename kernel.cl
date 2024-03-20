
#define BLOCK_SIZE 16

__kernel void mmul(
                const unsigned int             N,
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict local_A,
                __local        float* restrict local_B)
{
  float temp = 0.0f;
  int local_k, block_k;

  const int num_blocks = N/BLOCK_SIZE;


  const int i = get_global_id(0);
  const int j = get_global_id(1);


  const int block_i = get_group_id(0);
  const int block_j = get_group_id(1);

  const int local_i = get_local_id(0);
  const int local_j = get_local_id(1);


  int base_A = block_j*N*BLOCK_SIZE;    
  const int inc_A  = BLOCK_SIZE;

  int base_B = block_i*BLOCK_SIZE;
  const int inc_B  = BLOCK_SIZE*N;


  for(block_k = 0; block_k<num_blocks; block_k++)
    {
      local_A[local_j*BLOCK_SIZE+local_i] = A[base_A+local_j*N+local_i];
      local_B[local_j*BLOCK_SIZE+local_i] = B[base_B+local_j*N+local_i];

      barrier(CLK_LOCAL_MEM_FENCE);

      #pragma unroll
      for(local_k=0; local_k<BLOCK_SIZE; local_k++)
        temp += local_A[local_j*BLOCK_SIZE+local_k] * local_B[local_k*BLOCK_SIZE+local_i];

      barrier(CLK_LOCAL_MEM_FENCE);
      base_A += inc_A;
      base_B += inc_B;
    }

  C[j*N+i] = temp;

}