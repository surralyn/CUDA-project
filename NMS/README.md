# NMS on GPU
## "One shot" method 
In this implement: https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/nms/nms_kernel.cu , all pairs of IOU were calculatad from input boxes, which theoretically would be inefficient when input contains too many boxes. In fact, experiments show that it's significantly slower then PyTorch nms (CPU) when box number is larger than 10000.  

*This method is called "one shot" method in this file.
## "Iterate" method
The bottleneck of "one shot" is that first several boxes should filter out considerable number of boxes with lower score, and "one shot" does the calculation on them after all. It has time complexity of O(n^2). If we can filter most of the boxes using first several boxes, performance could be better.  
"nms_gpu.cu" introduces an iterate method to filter out following boxes using the first box. Besides, a simplified version of "one shot" method with the same performance is implemented.  
## Adaptive method
Do iterate in first several turns. Under certain rule, it automatically switch to "one shot". Four different methods (including PyTorch nms) performance are shown in the following figure: 

![experiment on nms](https://github.com/surralyn/CUDA-project/blob/master/NMS/exp.jpg)

This shows adaptive method always has 3~6 times faster performance than PyTorch nms.

## Demo
Generate input file in ./data dir:  

    python generate_data.py

Show PyTorch nms performance:  

    python nms_cpu.py

Show adaptive performance:  

    nvcc demo.cu -o demo && demo

## Experiment details
Hardware: Core i5 9400F & GeForce GTX 960 2G  
Number of boxes: 2^10 ~ 2^17  
IOU threshold: 0.1  
Image resolution: 1080p  
