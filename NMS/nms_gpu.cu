#include "../common/book.h"
#define CEILDIV(m,n) (((m) + (n) - 1) / (n))
typedef unsigned long long ULL;
const int ULL_SIZE = sizeof(ULL);
const int ULL_LEN = 8*sizeof(ULL);


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

__global__ void nms_iter(float *boxes_dev, float *area_dev, bool *mask_dev, int n, float th){
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1, box_idx = 5*idx;
    if(idx < n){
        float IOU = get_iou_dev(boxes_dev, boxes_dev + box_idx, area_dev[0], area_dev[idx]);
        if(IOU > th) mask_dev[idx] = false;
        else mask_dev[idx] = true;
    }
}

__global__ void nms_full(float *boxes_dev, float *area_dev, ULL *mask_dev, int n, float th){
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    if (row_start > col_start) return;

    const int row_size = min(n - row_start * blockDim.x, blockDim.x);
    const int col_size = min(n - col_start * blockDim.x, blockDim.x);

    __shared__ float block_boxes[ULL_LEN * 5];
    __shared__ float block_area[ULL_LEN];
    if (threadIdx.x < col_size) {
        int idx = blockDim.x * col_start + threadIdx.x;
        block_area[threadIdx.x] = area_dev[idx];
        block_boxes[threadIdx.x * 5 + 0] = boxes_dev[idx * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] = boxes_dev[idx * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] = boxes_dev[idx * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] = boxes_dev[idx * 5 + 3];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        int cur_box_idx = blockDim.x * row_start + threadIdx.x;
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

float inline area(float *boxes_host){
    float width = boxes_host[2] - boxes_host[0];
    float height = boxes_host[3] - boxes_host[1];
    if(width < 0 || height < 0) return 0;
    else return width * height;
}

float get_rate(int n0, int n1){
    int dn = n0 - n1;
    float alpha = 1.0 - 1.0 / float(dn);
    float r = float(dn) / float(n0);
    return alpha * r;
}

int nms_part_gpu(float* boxes_host, float th, int boxes_num){
    int block_size = 1024;
    float *area_dev=NULL, *boxes_dev=NULL;
    int area_size = boxes_num * sizeof(float);
    int box_size = 5 * area_size;
    int mask_size = boxes_num * sizeof(bool);
    bool *mask_dev=NULL, *mask_host=NULL;
    mask_host = (bool*) malloc(mask_size);
    HANDLE_ERROR(cudaMalloc((void **)&area_dev, area_size));
    HANDLE_ERROR(cudaMalloc((void **)&boxes_dev, box_size));
    HANDLE_ERROR(cudaMalloc((void **)&mask_dev, mask_size));

    int start = 0, k, cnt = 0;
    while(boxes_num){
        area_size = boxes_num * sizeof(float);
        box_size = 5 * area_size;
        mask_size = boxes_num * sizeof(bool);

        HANDLE_ERROR(cudaMemcpy(boxes_dev + 5*start, boxes_host + 5*start, box_size, cudaMemcpyHostToDevice));
        get_area<<<CEILDIV(boxes_num, block_size), block_size>>>(boxes_dev + 5*start, area_dev, boxes_num);
        if(cnt < 5 && boxes_num > 3000){
            nms_iter<<<CEILDIV(boxes_num, block_size), block_size>>> (boxes_dev + 5*start, area_dev, mask_dev, boxes_num, th);
            HANDLE_ERROR(cudaMemcpy(mask_host, mask_dev, mask_size, cudaMemcpyDeviceToHost));

            k = start;
            for(int i=1; i<boxes_num; i++){
                if(mask_host[i]){
                    k++;
                    for(int j=0;j<5;j++) boxes_host[5*k + j] = boxes_host[5*(start + i) + j];
                }
            }
            if(k > start && get_rate(boxes_num, k - start) < 0.01) cnt++;
            else cnt = 0;
            boxes_num = k - start;
            start++;
        }else{
            HANDLE_ERROR(cudaFree(mask_dev));
            int mask_piece_num = CEILDIV(boxes_num, ULL_LEN);
            int mask_size_piece = mask_piece_num * ULL_SIZE;
            ULL *mask_host_piece = (ULL*) malloc(mask_size_piece);
            ULL mask_size_full = boxes_num * mask_piece_num * ULL_SIZE;
            ULL *mask_host_full = (ULL*) malloc(mask_size_full), *mask_dev_full=NULL;
            HANDLE_ERROR(cudaMalloc((void **)&mask_dev_full, mask_size_full));
            
            dim3 blocks(mask_piece_num, mask_piece_num);
            nms_full<<<blocks, ULL_LEN>>>(boxes_dev + 5*start, area_dev, mask_dev_full, boxes_num, th);
            
            HANDLE_ERROR(cudaMemcpy(mask_host_full, mask_dev_full, mask_size_full, cudaMemcpyDeviceToHost));
            
            memset(mask_host_piece, 0ULL, mask_size_piece);
            k = start;
            for(int i = 0; i < boxes_num; i++){
                int piece_idx = i / ULL_LEN;
                int piece_bias = i % ULL_LEN;
                bool b = ! (mask_host_piece[piece_idx] & (1ULL << piece_bias));
                if(i == 0 || b){
                    for(int j=0;j<5;j++) boxes_host[5*k + j] = boxes_host[5*(start + i) + j];
                    k++;
                    ULL *p = mask_host_full + i * mask_piece_num;
                    for (int j = piece_idx; j < mask_piece_num; j++){
                        mask_host_piece[j] |= p[j];
                    }
                }
            }
            return k;
        }
    }
    return start;
}