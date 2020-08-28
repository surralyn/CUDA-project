#include "../common/book.h"
#define CEILDIV(m,n) (((m) + (n) - 1) / (n))
typedef long long LL;
typedef unsigned long long ULL;
const int ULL_SIZE = sizeof(ULL);
const int ULL_LEN = 8*ULL_SIZE;
#include "nms_kernel.cu"


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

__global__ void nms_iter(float *tar_box, float *tar_area, float *boxes_dev, float *area_dev, bool *mask_dev, int n, float th){
    int idx = blockIdx.x * blockDim.x + threadIdx.x, box_idx = 5*idx;
    if(idx < n){
        float IOU = get_iou_dev(tar_box, boxes_dev + box_idx, tar_area[0], area_dev[idx]);
        if(IOU > th) mask_dev[idx] = false;
        else mask_dev[idx] = true;
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
    float *area_dev=NULL, *boxes_dev=NULL;
    int area_size = boxes_num * sizeof(float), box_size = 5 * area_size, mask_size = boxes_num* sizeof(bool);
    int block_size = 256, full_part = 6000;
    bool *mask_dev=NULL, *mask_host=NULL;
    mask_host = (bool*) malloc(mask_size);
    HANDLE_ERROR(cudaMalloc((void **)&area_dev, area_size));
    HANDLE_ERROR(cudaMalloc((void **)&boxes_dev, box_size));
    HANDLE_ERROR(cudaMalloc((void **)&mask_dev, mask_size));

    int start = 0, k;
    while(boxes_num){
        area_size = boxes_num * sizeof(float);
        box_size = 5 * area_size;
        mask_size = boxes_num * sizeof(bool);

        HANDLE_ERROR(cudaMemcpy(boxes_dev + 5*start, boxes_host + 5*start, box_size, cudaMemcpyHostToDevice));
        get_area<<<CEILDIV(boxes_num, block_size), block_size>>>(boxes_dev + 5*start, area_dev, boxes_num);
        if(boxes_num > full_part){
            nms_iter<<<CEILDIV(boxes_num, block_size), block_size>>>
                    (boxes_dev + 5*start, area_dev, boxes_dev + 5*(start + 1), area_dev + 1, 
                    mask_dev, boxes_num - 1, th);
            HANDLE_ERROR(cudaMemcpy(mask_host, mask_dev, mask_size, cudaMemcpyDeviceToHost));
        }else{
            HANDLE_ERROR(cudaFree(mask_dev));
            int col_blocks = CEILDIV(boxes_num, threadsPerBlock);
            ULL mask_size_full = boxes_num * col_blocks * ULL_SIZE;
            ULL *mask_host_full = (ULL*) malloc(mask_size_full), *mask_dev_full=NULL;
            HANDLE_ERROR(cudaMalloc((void **)&mask_dev_full, mask_size_full));
            
            dim3 blocks(CEILDIV(boxes_num, threadsPerBlock),
                CEILDIV(boxes_num, threadsPerBlock));
            dim3 threads(threadsPerBlock);
            nms_full<<<blocks, threads>>>(boxes_dev + 5*start, area_dev, mask_dev_full, boxes_num, th);
            
            HANDLE_ERROR(cudaMemcpy(mask_host_full, mask_dev_full, mask_size_full, cudaMemcpyDeviceToHost));
            
            memset(mask_host, true, mask_size);
            for(int i=0;i<boxes_num;i++){
                if(mask_host[i]){
                    for(int j=i+1;j<boxes_num;j++){
                        int nblock = j / threadsPerBlock;
                        int iblock = j % threadsPerBlock;
                        ULL t = *(mask_host_full + i * col_blocks + nblock);
                        if(t & (1ULL << iblock)) mask_host[j] = false;
                    }
                }
            }
            mask_host++;
        }
        
        k = start;
        for(int i=1; i<boxes_num; i++){
            if(mask_host[i-1]){
                k++;
                for(int j=0;j<4;j++) boxes_host[5*k + j] = boxes_host[5*(start+i) +j];
            }
        }

        if(boxes_num > full_part){
            boxes_num = k - start;
            start++;
        }else return k + 1;
    }
    
    return start;
}