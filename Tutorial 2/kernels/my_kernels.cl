kernel void histo(global const uchar* A, global int* H)
//function used from Tutorial 3 of Parallel Programming
{
		int id = get_global_id(0);

		int binIndex = A[id];
		//assumes that H has been initialised to location zero 
		atomic_inc(&H[binIndex]);
		//this atomic operation computes the histogram and stores the values at their buckets
		//atomic operations are very inefficient
}

kernel void histoC(global const int* A, global int* cH, const int binSize)
{
		int id = get_global_id(0);

		for (int i = id + 1; i < binSize && id < binSize; i++) {
			atomic_add(&cH[i], A[id]/3);
		}
}

kernel void LUT(global const int* A, global int* B)
{
	int id = get_global_id(0);

	B[id] = A[id] * (double)255 / A[255];
}
