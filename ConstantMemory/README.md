# CUDA constant memory test

This project shows how constant memory accelerate calculation in a convolution layer.

## To Run a Demo

nvcc constantMemoryTest.cu -o constantMemoryTest && constantMemoryTest  

Control parameters in "constantMemoryTest.cu":  
* outChannel: If you want to set this parameter larger than 16, make sure to change the other 2 parameters in constantMemoryTest.cu (line 92), since the block size is defined to be bsx * bsy * outChannel and shouldn't ba larger than 1024.  
* errorTolerance: While checking GPU results, this will be using as a threshold to determine whether a single pixel is wrong.  

Results on Geforce GTX 1660Ti:  

CUDA using constant memory test.  

Initial parameters:  
inChannel=3, outChannel=8, height=1080, width=1920, kernelSize=5  

Calculating on cpu...  
CPU calculating done.(time consuming: 4545.302ms)  

Calculating on gpu...  
GPU calculating done.(time consuming: 7.952ms)  
Result check: error rate = 0.0000 (ratio of abs error more than 0.000010)  

Calculating on gpu (with constant memory)...  
GPU calculating done (with constant memory).(time consuming: 5.806ms)  
Result check: error rate (with constant memory) = 0.0000 (ratio of abs error more than 0.000010)  