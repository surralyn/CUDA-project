#include "../common/book.h"
#include <windows.h>
#include "utils.h"
#include <stdio.h>

const int inChannel=3, outChannel=8, height=1080, width=1920, kernelSize=5, 
			outputHeight = height - kernelSize + 1, outputWidth = width - kernelSize + 1;
const float errorTolerance = 1e-5;
__constant__ float kernelDeviceConstant[inChannel * outChannel * kernelSize * kernelSize];

__global__ void conv(float *featureMap, float *kernel, float *output){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    if(i < outputHeight && j < outputWidth && k < outChannel){
        int idx = k*outputHeight*outputWidth + j*outputHeight + i;
        int h, w;
        float temp=0;
        for(int m=0;m<kernelSize;m++){
            for(int n=0;n<kernelSize;n++){
                h = i+m; w = j+n;
                for(int p=0;p<inChannel;p++){
                    temp +=
                        kernel[k*kernelSize*kernelSize*inChannel + p*kernelSize*kernelSize + n*kernelSize + m] * 
                        featureMap[p*height*width + w*height + h];
                }
            }
        }
        output[idx] = temp;
    }
}

__global__ void conv(float *featureMap, float *output){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    if(i < outputHeight && j < outputWidth && k < outChannel){
        int idx = k*outputHeight*outputWidth + j*outputHeight + i;
        int h, w;
        float temp=0;
        for(int m=0;m<kernelSize;m++){
            for(int n=0;n<kernelSize;n++){
                h = i+m; w = j+n;
                for(int p=0;p<inChannel;p++){
                    temp +=
                        kernelDeviceConstant[k*kernelSize*kernelSize*inChannel + p*kernelSize*kernelSize + n*kernelSize + m] * 
                        featureMap[p*height*width + w*height + h];
                }
            }
        }
        output[idx] = temp;
    }
}

int main(){
    printf("\nCUDA using constant memory test.\n");
    printf("\nInitial parameters:\ninChannel=%d, outChannel=%d, height=%d, width=%d, kernelSize=%d\n", 
                                    inChannel, outChannel, height, width, kernelSize);
    int sizeFeatureMap = inChannel * height * width * sizeof(float), 
        sizeKernel = inChannel * kernelSize * kernelSize * outChannel * sizeof(float), 
        sizeOutput = outChannel * outputHeight * outputWidth * sizeof(float);
    float *featureMap, *kernel, *featureMapDevice, *kernelDevice, *output, *outputDevice, *outputHost, dt, errorRate;
	//malloc:
    featureMap = (float *) malloc(sizeFeatureMap);
    kernel = (float *) malloc(sizeKernel);
    output = (float *) malloc(sizeOutput);
	outputHost = (float *) malloc(sizeOutput);
    HANDLE_ERROR(cudaMalloc((void **)&featureMapDevice, sizeFeatureMap));
    HANDLE_ERROR(cudaMalloc((void **)&kernelDevice, sizeKernel));
    HANDLE_ERROR(cudaMalloc((void **)&outputDevice, sizeOutput));
    //prepare data:
    initMap(featureMap, inChannel, height, width);
    initKernel(kernel, inChannel, outChannel, kernelSize);
    
    //load data:
    HANDLE_ERROR(cudaMemcpy(featureMapDevice, featureMap, sizeFeatureMap, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(kernelDevice, kernel, sizeKernel, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(kernelDeviceConstant, kernel, sizeKernel));
    //Event:
    cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
    //get cpu result:
    printf("\nCalculating on cpu...\n");
    dt = get_time();

    convHost(featureMap, kernel, output, inChannel, outChannel, height, width, kernelSize, outputHeight, outputWidth);

    dt = get_time() - dt;
	printf("CPU calculating done.(time consuming: %.3fms)\n", dt);
    //get gpu result:
    int bsx=8, bsy=8, bsz=outChannel;
    dim3 gridSize((height + bsx - 1) / bsx, (width + bsy - 1) / bsy, 1);
    dim3 blockSize(bsx, bsy, bsz);
	cudaThreadSynchronize();
    printf("\nCalculating on gpu...\n");
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    conv<<<gridSize, blockSize>>>(featureMapDevice, kernelDevice, outputDevice);

    cudaThreadSynchronize();
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &dt, start, stop ) );
	printf("GPU calculating done.(time consuming: %.3fms)\n", dt);

    HANDLE_ERROR(cudaMemcpy(outputHost, outputDevice, sizeOutput, cudaMemcpyDeviceToHost));
	
	errorRate = compareMap(output, outputHost, outputHeight, outputWidth, outChannel, errorTolerance);
    printf("Result check: error rate = %.4f (ratio of abs error more than %f)\n", errorRate, errorTolerance);

    //get gpu result with constant memory:
    cudaThreadSynchronize();
    printf("\nCalculating on gpu (with constant memory)...\n");
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    conv<<<gridSize, blockSize>>>(featureMapDevice, outputDevice);
	
    cudaThreadSynchronize();
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &dt, start, stop ) );
	printf("GPU calculating done (with constant memory).(time consuming: %.3fms)\n", dt);

    HANDLE_ERROR(cudaMemcpy(outputHost, outputDevice, sizeOutput, cudaMemcpyDeviceToHost));
	
	errorRate = compareMap(output, outputHost, outputHeight, outputWidth, outChannel, errorTolerance);
    printf("Result check: error rate (with constant memory) = %.4f (ratio of abs error more than %f)\n", errorRate, errorTolerance);

    cudaFree(featureMapDevice);
    cudaFree(kernelDevice);
    cudaFree(outputDevice);
    free(featureMap);
    free(kernel);
    free(output);
    free(outputHost);
    HANDLE_ERROR( cudaEventDestroy( start ) );
 	HANDLE_ERROR( cudaEventDestroy( stop ) );
    return 0;
}