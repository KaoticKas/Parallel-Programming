kernel void histo(global const unchar* A, global int* H)
//function used from Tutorial 3 of Parallel Programming
{
		int id = get_global_id(0)

		int binIndex =[id]
		//assumes that H has been initialised to location zero 
		atomic_inc(&H[binIndex])
		//this atomic operation computes the histogram and stores the values at their buckets
		//atomic operations are very inefficient
}

kernel void histoC(global int* A, global int* B)
{
		int id = get_global_id(0);
		int gsize = get_global_size(0);

		for (int i = id + 1; i < gsize; i++) {
			atomic_add(&B[i],A[id])
		}

}
