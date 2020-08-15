template <unsigned int blockSize>
__global__ void reduce_final(int *g_idata, int *g_odata, int num) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;
	unsigned int gridSize = blockDim.x * gridDim.x;
	sdata[tid] = 0;
	while(i < num){
		sdata[tid] += g_idata[i];
		i += gridSize;
	}
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


template <void(*reduction_method)(int *g_idata, int *g_odata, int num)>
void test_final(int num, int bs, int reduce_gs=1){
	int gs = (num + bs - 1) / bs / reduce_gs;
	int *i_data = NULL, *o_data = NULL, *g_idata = NULL, *g_odata = NULL, r1=0, r2=0;
	
	//malloc:
	i_data = (int *) malloc(num * sizeof(int));
	o_data = (int *) malloc(gs * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**)&g_idata, num*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&g_odata, gs*sizeof(int)));
	
	//prepare data:
	for(int i=0;i<num;i++)i_data[i]= i % 3;
	HANDLE_ERROR(cudaMemcpy(g_idata, i_data, num*sizeof(int), cudaMemcpyHostToDevice));
	
	//GPU calculate:
	double dt;
	cudaThreadSynchronize();
	warmup();
	dt = get_time();
	
	reduction_method<<<gs, bs, bs*sizeof(int)>>>(g_idata, g_odata, num);
	
	cudaThreadSynchronize();
	dt = get_time() - dt;
	
	unsigned long long dclock = clock();
	
	
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