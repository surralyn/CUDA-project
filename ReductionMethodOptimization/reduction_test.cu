#include <stdio.h>
#include "book.h"
#include "reduction_methods.h"
#include "finalReductionMethod.h"


int main(){
	int num = (1<<28) + 101, bs = 1<<10, pre_add = 8;
	printf("\nMethod 1 ---- Interleaved Addressing with warp divergence:\n");
	test<reduce1>(num, bs);
	printf("\nMethod 2 ---- Interleaved Addressing with bank conflict:\n");
	test<reduce2>(num, bs);
	printf("\nMethod 3 ---- Sequential Addressing:\n");
	test<reduce3>(num, bs);
	printf("\nMethod 4 ---- First Add During Load:\n");
	test<reduce4>(num, bs, 2);
	printf("\nMethod 5 ---- Unroll the Last Warp:\n");
	test<reduce5>(num, bs, 2);
	printf("\nMethod 6 ---- Completely Unrolled:\n");
	switch(bs){
		case 1024:
			test<reduce6<1024>>(num, bs, 2);break;
		case 512:
			test<reduce6<512>>(num, bs, 2);break;
		case 256:
			test<reduce6<256>>(num, bs, 2);break;
		case 128:
			test<reduce6<128>>(num, bs, 2);break;
		case 64:
			test<reduce6<64>>(num, bs, 2);break;
		case 32:
			test<reduce6<32>>(num, bs, 2);break;
		case 16:
			test<reduce6<16>>(num, bs, 2);break;
		case 8:
			test<reduce6<8>>(num, bs, 2);break;
		case 4:
			test<reduce6<4>>(num, bs, 2);break;
		case 2:
			test<reduce6<2>>(num, bs, 2);break;
		case 1:
			test<reduce6<1>>(num, bs, 2);break;
	}
	printf("\nMethod final ---- Multiple Adds:\n");
	switch(bs){
		case 1024:
			test_final<reduce_final<1024>>(num, bs, pre_add);break;
		case 512:
			test_final<reduce_final<512>>(num, bs, pre_add);break;
		case 256:
			test_final<reduce_final<256>>(num, bs, pre_add);break;
		case 128:
			test_final<reduce_final<128>>(num, bs, pre_add);break;
		case 64:
			test_final<reduce_final<64>>(num, bs, pre_add);break;
		case 32:
			test_final<reduce_final<32>>(num, bs, pre_add);break;
		case 16:
			test_final<reduce_final<16>>(num, bs, pre_add);break;
		case 8:
			test_final<reduce_final<8>>(num, bs, pre_add);break;
		case 4:
			test_final<reduce_final<4>>(num, bs, pre_add);break;
		case 2:
			test_final<reduce_final<2>>(num, bs, pre_add);break;
		case 1:
			test_final<reduce_final<1>>(num, bs, pre_add);break;
	}
	cudaDeviceReset();
	return 0;
}