#include <stdio.h>
#include <cuda_runtime.h>
int main(){
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("Free memory: %lu bytes\n", free_mem);
printf("Total memory: %lu bytes\n", total_mem);
return 0;
}