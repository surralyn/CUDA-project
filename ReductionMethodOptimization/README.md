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
time consuming: 43.021ms  
bandwidth: 24.96 GB/s  

Method 2 ---- Interleaved Addressing with bank conflict:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 30.175ms  
bandwidth: 35.58 GB/s  

Method 3 ---- Sequential Addressing:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 29.002ms  
bandwidth: 37.02 GB/s  

Method 4 ---- First Add During Load:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 13.913ms  
bandwidth: 77.17 GB/s  

Method 5 ---- Unroll the Last Warp:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 10.498ms  
bandwidth: 102.28 GB/s  

Method 6 ---- Completely Unrolled:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 7.723ms  
bandwidth: 139.03 GB/s  

Method final ---- Multiple Adds:  
GPU result: 268435557 (actual: 268435557)  correct: 1  
time consuming: 4.972ms  
bandwidth: 215.97 GB/s  