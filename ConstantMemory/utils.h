
double get_time(void)
{
    LARGE_INTEGER timer;
    static LARGE_INTEGER fre;
    static int init = 0;
    double t;

    if (init != 1) {
        QueryPerformanceFrequency(&fre);
        init = 1;
    }

    QueryPerformanceCounter(&timer);

    t = timer.QuadPart * 1000.0 / fre.QuadPart;

    return t;
}

float randFloat(){
    int randInt = rand() % 1000;
    return randInt / 1000.0;
}

void initMap(float *featureMap, int inChannel, int height, int width){
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            for(int k=0;k<inChannel;k++){
                featureMap[k*height*width + j*height + i] = randFloat();
            }
        }
    }
}

void initKernel(float *kernel, int inChannel, int outChannel, int kernelSize){
    for(int i=0;i<kernelSize;i++){
        for(int j=0;j<kernelSize;j++){
            for(int k=0;k<inChannel;k++){
                for(int l=0;l<outChannel;l++){
                    kernel[l*kernelSize*kernelSize*inChannel + k*kernelSize*kernelSize + j*kernelSize + i] = randFloat();
                }
            }
        }
    }
}

void convHost(float *featureMap, float *kernel, float *output, 
    int inChannel, int outChannel, int height, int width, int kernelSize, int outputHeight, int outputWidth){
        int h, w;
        for(int i=0;i<outputHeight;i++){
            for(int j=0;j<outputWidth;j++){
                for(int k=0;k<outChannel;k++){
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
                    output[k*outputHeight*outputWidth + j*outputHeight + i] = temp;
                }
            }
        }
    }

void showMap(float *featureMap, int height, int width, int channel){
    for(int k=0;k<channel;k++){
        printf("Channel %d:\n", k+1);
        for(int j=0;j<width;j++){
            for(int i=0;i<height;i++){
                printf("%.2f ", featureMap[k*height*width + j*height + i]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

void showKernel(float *kernel, int kernelSize, int inChannel, int outChannel){
    for(int l=0;l<outChannel;l++){
		printf("Kenel %d:\n", l+1);
		for(int k=0;k<inChannel;k++){
			printf("Channel %d:\n", k+1);
			for(int j=0;j<kernelSize;j++){
				for(int i=0;i<kernelSize;i++){
					printf("%.2f ", kernel[l*kernelSize*kernelSize*inChannel + k*kernelSize*kernelSize + j*kernelSize + i]);
				}
				printf("\n");
			}
		}
		printf("\n");
	}
}

float compareMap(float *CPUfeatureMap, float *GPUfeatureMap, int height, int width, int channel, float eps){
    float error;
	int r=0;
    for(int k=0;k<channel;k++){
        for(int j=0;j<width;j++){
            for(int i=0;i<height;i++){
                int idx = k*height*width + j*height + i;
				error = abs(CPUfeatureMap[idx] - GPUfeatureMap[idx]);
                if(error > eps){
					if(r<10)printf("%.6f\n", error);
					r++;
				}
            }
        }
    }
    return 1.0 * r / height / width / channel;
}