__global__ void reduce1(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;
	sdata[tid] = g_idata[i];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s=1;s<blockDim.x;s*=2){
		if(tid % (2*s) == 0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce2(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;
	sdata[tid] = g_idata[i];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s=1;s<blockDim.x;s*=2){
		int idx = tid * s * 2;
		if(idx < blockDim.x){
			sdata[idx] += sdata[idx + s];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce3(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;
	sdata[tid] = g_idata[i];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce4(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + tid;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}


__device__ void warpReduce(volatile int *sdata, int tid){
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce5(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + tid;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2;s>32;s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if(tid < 32)warpReduce(sdata, tid);
	// write result for this block to global mem
	if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__device__ void totalWarpReduce(volatile int *sdata, int tid){
	if(blockSize >= 64)sdata[tid] += sdata[tid + 32];
	if(blockSize >= 32)sdata[tid] += sdata[tid + 16];
	if(blockSize >= 16)sdata[tid] += sdata[tid + 8];
	if(blockSize >= 8)sdata[tid] += sdata[tid + 4];
	if(blockSize >= 4)sdata[tid] += sdata[tid + 2];
	if(blockSize >= 2)sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + tid;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	
	// do reduction in shared mem
	if(blockSize >= 1024){
		if(tid < 512)sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if(blockSize >= 512){
		if(tid < 256)sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128)sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64)sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	
	if(tid < 32)totalWarpReduce<blockSize>(sdata, tid);
	// write result for this block to global mem
	if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}

template <void(*reduction_method)(int *g_idata, int *g_odata, int num)> void test(int num, int bs, int reduce_gs=1){}
template <void(*reduction_method)(int *g_idata, int *g_odata)>
void test(int num, int bs, int reduce_gs=1){
	int gs = (num + bs - 1) / bs;
	gs = (gs + reduce_gs - 1) / reduce_gs;
	int aloc_size = gs * reduce_gs * bs;
	int *i_data = NULL, *o_data = NULL, *g_idata = NULL, *g_odata = NULL, r1=0, r2=0;
	
	//malloc:
	i_data = (int *) malloc(aloc_size * sizeof(int));
	o_data = (int *) malloc(gs * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**)&g_idata, aloc_size*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&g_odata, gs*sizeof(int)));
	
	//prepare data:
	for(int i=0;i<num;i++)i_data[i]= i % 3;
	HANDLE_ERROR(cudaMemcpy(g_idata, i_data, aloc_size*sizeof(int), cudaMemcpyHostToDevice));
	
	//GPU calculate:
	double dt;
	cudaThreadSynchronize();
	warmup();
	dt = get_time();
	
	reduction_method<<<gs, bs, bs*sizeof(int)>>>(g_idata, g_odata);
	
	cudaThreadSynchronize();
	dt = get_time() - dt;
	
	//CPU calculate:
	HANDLE_ERROR(cudaMemcpy(o_data, g_odata, gs*sizeof(int), cudaMemcpyDeviceToHost));
	for(int i=0;i<gs;i++)r1 += o_data[i];
	for(int i=0;i<num;i++)r2 += i_data[i];
	//show result:
	printf("GPU result: %d (actual: %d)  correct: %d\n", r1, r2, r1==r2);
	printf("time consuming: %.3lfms\n", 1000*dt);
	printf("bandwidth: %.2f GB/s\n", num/dt*4/1e9);
	
	//
	cudaFree(g_idata);
	cudaFree(g_odata);
	free(i_data);
	free(o_data);
}