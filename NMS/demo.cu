#include <stdio.h>
#include <time.h>
#include "nms_kernel.cu"
#include "nms_part_gpu.cu"


int main(){
    FILE *fp = NULL;
    fp = fopen("./data/input.txt", "r");
    int *keep_out=NULL, *num_out=NULL, boxes_num, boxes_dim=5, device_id=0;
    float* boxes_host, *pure_boxes_host, dt;
    float nms_overlap_thresh;
    fscanf(fp, "%d %f", &boxes_num, &nms_overlap_thresh);
    boxes_host = (float*) malloc(5 * boxes_num * sizeof(float));
    pure_boxes_host = (float*) malloc(4 * boxes_num * sizeof(float));
    keep_out = (int*) malloc(boxes_num * sizeof(int));
    num_out = (int*) malloc(sizeof(int));
    for(int i=0;i<boxes_num;i++){
        for(int j=0;j<5;j++)fscanf(fp, "%f", &boxes_host[5*i + j]);
        for(int j=0;j<4;j++)pure_boxes_host[4*i + j] = boxes_host[5*i + j];
    }
    cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    _nms(keep_out, num_out, boxes_host, boxes_num, boxes_dim, nms_overlap_thresh, device_id);
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &dt, start, stop ) );
    printf("Open source coda:\nTime consuming: %fms\n", dt);
    
    printf("Remaining box num: %d\n", *num_out);

    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    int k=nms_part_gpu(pure_boxes_host, nms_overlap_thresh, boxes_num);
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &dt, start, stop ) );
    printf("\nMy method:\nTime consuming: %fms\n", dt);
    printf("Remaining box num: %d\n", k);

    fclose(fp);
    HANDLE_ERROR( cudaEventDestroy( start ) );
 	HANDLE_ERROR( cudaEventDestroy( stop ) );
    return 0;
}