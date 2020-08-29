#include "../common/book.h"
#define CEILDIV(m,n) (((m) + (n) - 1) / (n))
#define ULL_LEN threadsPerBlock
typedef unsigned long long ULL;
const int ULL_SIZE = sizeof(ULL);
const int threadsPerBlock = 8*sizeof(ULL);


__global__ void get_area(float *boxes_dev, float *area_dev, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x, box_idx = 5*idx;
    if(idx < n){
        area_dev[idx] = (boxes_dev[box_idx+2] - boxes_dev[box_idx]) * 
                        (boxes_dev[box_idx+3] - boxes_dev[box_idx+1]);
    }
}

__device__ inline float get_iou_dev(float *a, float *b, float Sa, float Sb) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    if(right < left || bottom < top) return 0;
    float inter = (right - left) * (bottom - top);
    return inter / (Sa + Sb - inter);
}

__global__ void nms_iter(float *boxes_dev, float *area_dev, ULL *mask_dev, int n, float th){
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1, box_idx = 5*idx;
    if(idx < n){
        float IOU = get_iou_dev(boxes_dev, boxes_dev + box_idx, area_dev[0], area_dev[idx]);
        idx--;
        if(IOU > th) atomicOr(mask_dev + idx / ULL_LEN, 1ULL << (idx % ULL_LEN));
    }
}

inline float get_inter_host(float const * const a, float const * const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    if(right < left || bottom < top) return 0;
    return (right - left) * (bottom - top);
}

int nms_cpu(float *boxes_host, float *area_host, bool *mask_host, int n, float th){
    int idx, num = 0;
    float inter, IOU;
    for(int i=0;i<n;i++){
        if(mask_host[i]){
            idx = 5*i;
            for(int j=i+1;j<n;j++){
                if(mask_host[j]){
                    inter = get_inter_host(boxes_host + idx, boxes_host + 5*j);
                    IOU = inter / (area_host[i] + area_host[j] - inter);
                    if(IOU > th)mask_host[j] = false;
                }
            }
        }
    }
    for(int i=0;i<n;i++){
        if(mask_host[i]){
            for(int j=0;j<5;j++) boxes_host[5*num + j] = boxes_host[5*i + j];
            num++;
        }
    }
    return num;
}



__global__ void nms_full(float *boxes_dev, float *area_dev, ULL *mask_dev, int n, float th){
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    if (row_start > col_start) return;

    const int row_size = min(n - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = min(n - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 5];
    __shared__ float block_area[threadsPerBlock];
    if (threadIdx.x < col_size) {
        int idx = threadsPerBlock * col_start + threadIdx.x;
        block_area[threadIdx.x] = area_dev[idx];
        block_boxes[threadIdx.x * 5 + 0] = boxes_dev[idx * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] = boxes_dev[idx * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] = boxes_dev[idx * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] = boxes_dev[idx * 5 + 3];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        float *cur_box = boxes_dev + cur_box_idx * 5;
        int i = 0;
        ULL t = 0;
        int start = 0;
        if (row_start == col_start) start = threadIdx.x + 1;
        for (i = start; i < col_size; i++) {
            float IOU = get_iou_dev(cur_box, block_boxes + i * 5, 
                            area_dev[cur_box_idx], block_area[i]);
            if (IOU > th) t |= 1ULL << i;
        }
        mask_dev[cur_box_idx * gridDim.x + col_start] = t;
    }
}

int nms_part_gpu(float* boxes_host, float th, int boxes_num){
    int full_part = 6000;
    int block_size = 256;
    float *area_dev=NULL, *boxes_dev=NULL;
    int area_size = boxes_num * sizeof(float);
    int box_size = 5 * area_size;
    int mask_piece_num = CEILDIV(boxes_num, ULL_LEN);
    int mask_size = mask_piece_num * ULL_SIZE;
    ULL *mask_dev=NULL, *mask_host=NULL;
    mask_host = (ULL*) malloc(mask_size);
    HANDLE_ERROR(cudaMalloc((void **)&area_dev, area_size));
    HANDLE_ERROR(cudaMalloc((void **)&boxes_dev, box_size));
    HANDLE_ERROR(cudaMalloc((void **)&mask_dev, mask_size));

    int start = 0, k;
    while(boxes_num){
        area_size = boxes_num * sizeof(float);
        box_size = 5 * area_size;
        mask_piece_num = CEILDIV(boxes_num, ULL_LEN);
        mask_size = mask_piece_num * ULL_SIZE;

        HANDLE_ERROR(cudaMemcpy(boxes_dev + 5*start, boxes_host + 5*start, box_size, cudaMemcpyHostToDevice));
        get_area<<<CEILDIV(boxes_num, block_size), block_size>>>(boxes_dev + 5*start, area_dev, boxes_num);
        if(boxes_num > full_part){
            HANDLE_ERROR(cudaMemset(mask_dev, 0ULL, mask_size));
            nms_iter<<<CEILDIV(boxes_num, block_size), block_size>>> (boxes_dev + 5*start, area_dev, mask_dev, boxes_num, th);
            HANDLE_ERROR(cudaMemcpy(mask_host, mask_dev, mask_size, cudaMemcpyDeviceToHost));
        }else{
            HANDLE_ERROR(cudaFree(mask_dev));
            ULL mask_size_full = boxes_num * mask_piece_num * ULL_SIZE;
            ULL *mask_host_full = (ULL*) malloc(mask_size_full), *mask_dev_full=NULL;
            HANDLE_ERROR(cudaMalloc((void **)&mask_dev_full, mask_size_full));
            
            dim3 blocks(CEILDIV(boxes_num, threadsPerBlock),
                CEILDIV(boxes_num, threadsPerBlock));
            dim3 threads(threadsPerBlock);
            nms_full<<<blocks, threads>>>(boxes_dev + 5*start, area_dev, mask_dev_full, boxes_num, th);
            
            HANDLE_ERROR(cudaMemcpy(mask_host_full, mask_dev_full, mask_size_full, cudaMemcpyDeviceToHost));
            
            memset(mask_host, 0ULL, mask_size);

            for(int i = 0; i < boxes_num; i++){
                int piece_idx = i / ULL_LEN;
                int piece_bias = i % ULL_LEN;
                bool b = ! (mask_host[piece_idx] & (1ULL << piece_bias));
                if(i == 0 || b){
                    ULL *p = mask_host_full + i * mask_piece_num;
                    for (int j = piece_idx; j < mask_piece_num; j++){
                        mask_host[j] |= p[j];
                    }
                }
            }
        }
        
        k = start;
        for(int i=0; i<boxes_num - 1; i++){
            if(! (mask_host[i / ULL_LEN] & (1ULL << (i % ULL_LEN)))){
                k++;
                for(int j=0;j<4;j++) boxes_host[5*k + j] = boxes_host[5*(start + i + 1) + j];
            }
        }

        if(boxes_num > full_part){
            boxes_num = k - start;
            start++;
        }else return k;
    }
    
    return start;
}