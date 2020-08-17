#include <stdio.h>
#include "../common/book.h"
#include "reduction_methods.h"


int main(){
	int num = (1<<28) + 101, bs = 1<<10, pre_add = 8;

	printf("\nMethod 1 ---- Interleaved Addressing with warp divergence:\n");
	test(num, bs, reduce1);

	printf("\nMethod 2 ---- Interleaved Addressing with bank conflict:\n");
	test(num, bs, reduce2);

	printf("\nMethod 3 ---- Sequential Addressing:\n");
	test(num, bs, reduce3);

	printf("\nMethod 4 ---- First Add During Load:\n");
	test(num, bs, reduce4, 2);

	printf("\nMethod 5 ---- Unroll the Last Warp:\n");
	test(num, bs, reduce5, 2);

	printf("\nMethod 6 ---- Completely Unrolled:\n");
	switch(bs){
		case 1024:
			test(num, bs, reduce6<1024>, 2);break;
		case 512:
			test(num, bs, reduce6<512>, 2);break;
		case 256:
			test(num, bs, reduce6<256>, 2);break;
		case 128:
			test(num, bs, reduce6<128>, 2);break;
		case 64:
			test(num, bs, reduce6<64>, 2);break;
		case 32:
			test(num, bs, reduce6<32>, 2);break;
		case 16:
			test(num, bs, reduce6<16>, 2);break;
		case 8:
			test(num, bs, reduce6<8>, 2);break;
		case 4:
			test(num, bs, reduce6<4>, 2);break;
		case 2:
			test(num, bs, reduce6<2>, 2);break;
		case 1:
			test(num, bs, reduce6<1>, 2);break;
	}

	printf("\nMethod final ---- Multiple Adds:\n");
	switch(bs){
		case 1024:
			test(num, bs, NULL, pre_add, reduce_final<1024>);break;
		case 512:
			test(num, bs, NULL, pre_add, reduce_final<512>);break;
		case 256:
			test(num, bs, NULL, pre_add, reduce_final<256>);break;
		case 128:
			test(num, bs, NULL, pre_add, reduce_final<128>);break;
		case 64:
			test(num, bs, NULL, pre_add, reduce_final<64>);break;
		case 32:
			test(num, bs, NULL, pre_add, reduce_final<32>);break;
		case 16:
			test(num, bs, NULL, pre_add, reduce_final<16>);break;
		case 8:
			test(num, bs, NULL, pre_add, reduce_final<8>);break;
		case 4:
			test(num, bs, NULL, pre_add, reduce_final<4>);break;
		case 2:
			test(num, bs, NULL, pre_add, reduce_final<2>);break;
		case 1:
			test(num, bs, NULL, pre_add, reduce_final<1>);break;
	}

	cudaDeviceReset();
	return 0;
}