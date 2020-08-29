#include <stdio.h>
#include <time.h>
#include "nms_part_gpu.cu"


int main(){
    FILE *fp = NULL;
    fp = fopen("./data/input.txt", "r");
    int boxes_num;
    float* boxes_host, dt;
    float nms_overlap_thresh;
    fscanf(fp, "%d %f", &boxes_num, &nms_overlap_thresh);
    boxes_host = (float*) malloc(5 * boxes_num * sizeof(float));
    for(int i=0;i<boxes_num;i++){
        for(int j=0;j<5;j++)fscanf(fp, "%f", &boxes_host[5*i + j]);
    }
    cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );

    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    int k=nms_part_gpu(boxes_host, nms_overlap_thresh, boxes_num);
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