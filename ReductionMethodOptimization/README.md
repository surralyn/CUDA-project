# Reduction Optimization

This is a project based on https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf. According to the tutorial, there are 7 ways to optimize a reduction algrithm. This project accomplished those implements.

## To Run a Demo

nvcc reduction_test.cu -o reduction_test
./reduction_test

control parameters in "reduction_test.cu":
* num : length of the vector (defaut: (1<<28) + 101)
* bs : block size (defalt: 1024)
* pre_add: used in the final optimization, means gird size shrink scale (defaut: 8)

Results on Geforce GTX 1660Ti:

Method 1 ---- Interleaved Addressing with warp divergence:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 43.007ms  
bandwidth: 24.97 GB/s  

Method 2 ---- Interleaved Addressing with bank conflict:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 25.284ms  
bandwidth: 42.47 GB/s  

Method 3 ---- Sequential Addressing:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 26.768ms  
bandwidth: 40.11 GB/s  

Method 4 ---- First Add During Load:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 13.908ms  
bandwidth: 77.21 GB/s  

Method 5 ---- Unroll the Last Warp:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 9.562ms  
bandwidth: 112.29 GB/s  

Method 6 ---- Completely Unrolled:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 8.308ms  
bandwidth: 129.24 GB/s  

Method final ---- Multiple Adds:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 5.506ms  
bandwidth: 195.00 GB/s  