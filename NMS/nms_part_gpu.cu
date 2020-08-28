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

__device__ inline float get_inter_dev(float const * const a, float const * const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    if(right < left || bottom < top) return 0;
    return (right - left) * (bottom - top);
}

__global__ void nms_iter(float *tar_box, float *tar_area, float *boxes_dev, float *area_dev, bool *mask_dev, int n, float th){
    int idx = blockIdx.x * blockDim.x + threadIdx.x, box_idx = 5*idx;
    if(idx < n){
        float inter = get_inter_dev(tar_box, boxes_dev + box_idx);
        float IOU = inter / (tar_area[0] + area_dev[idx] - inter);
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

__device__ inline int revK(LL k){
	double t = sqrt(double(1 + 8*k));
	t = (t + 1) / 2;
	return LL(t);
}

__device__ void inline write_dev(ULL *can, ULL k){
    ULL q = k / ULL_LEN;
    int r = (ULL_LEN - 1 - k % ULL_LEN);
    atomicOr(can + q, 1ULL << r);
}

__global__ void nms_full(float *boxes_dev, float *area_dev, ULL *mask_dev, LL m, float th){
    LL idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < m){
        LL tar = revK(idx), cur = idx - tar * (tar-1) / 2;
        float inter = get_inter_dev(boxes_dev + 5*tar, boxes_dev + 5*cur);
        float IOU = inter / (area_dev[tar] + area_dev[cur] - inter);
        if(IOU > th) write_dev(mask_dev, idx);
    }
    
}

bool inline read_host(ULL *can, ULL k){
    ULL q = k / ULL_LEN;
    int r = (ULL_LEN - 1 - k % ULL_LEN);
    return can[q] & (1ULL << r);
}

void inline write_host(ULL *can, ULL k){
    ULL q = k / ULL_LEN;
    int r = (ULL_LEN - 1 - k % ULL_LEN);
    can[q] |= 1ULL << r;
}

int nms_part_gpu(float* boxes_host, float th, int boxes_num){
    float *area_dev=NULL, *boxes_dev=NULL;
    int area_size = boxes_num * sizeof(float), box_size = 5 * area_size, mask_size = boxes_num* sizeof(bool);
    int block_size = 256, full_part = -1;
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
            LL m = LL(boxes_num);
            m = m * (m - 1) / 2;
            LL mask_size_full = CEILDIV(m, ULL_SIZE);
            get_area<<<CEILDIV(boxes_num, block_size), block_size>>>(boxes_dev + 5*start, area_dev, boxes_num);
            ULL *mask_host_full = (ULL*) malloc(mask_size_full), *mask_dev_full=NULL;
            memset(mask_host_full, 0ULL, mask_size_full);
            HANDLE_ERROR(cudaFree(mask_dev));
            HANDLE_ERROR(cudaMalloc((void **)&mask_dev_full, mask_size_full));
            HANDLE_ERROR(cudaMemcpy(mask_dev_full, mask_host_full, mask_size_full, cudaMemcpyHostToDevice));
            nms_full<<<CEILDIV(m, block_size), block_size>>>(boxes_dev + 5*start, area_dev, mask_dev_full, m, th);
            HANDLE_ERROR(cudaMemcpy(mask_host_full, mask_dev_full, mask_size_full, cudaMemcpyDeviceToHost));
            memset(mask_host, true, mask_size);
            for(int i=0;i<boxes_num-1;i++){
                if(mask_host[i]){
                    for(int j=i+1;j<boxes_num;j++){
                        if(read_host(mask_host_full, j*(j-1)/2 + i)) mask_host[j] = false;
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