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
	//this function callculates the cummulative histogram by using pointers to vectors
	//and loops through until it reached the binsize
		int id = get_global_id(0);

		for (int i = id + 1; i < binSize && id < binSize; i++) {
			atomic_add(&cH[i], A[id]/3);
			//uses atomic function of add to compute the histogram that works with rgb and monochrome images.
		}
}

kernel void LUT(global const int* A, global int* B)
{
	//creates a lookup table with normalised values for each pixel
	int id = get_global_id(0);

	B[id] = A[id] * (double)255 / A[255];
	//b being the vector that stores the LUT values, normalises the intensities of the pixels
}

kernel void adjustImg(global uchar* A, global int* lut, global uchar* nImg) {
//this kernel adjusts the input image with the normalised histogram values from the look up table and
//casts them onto the image to produce the output image
	int id = get_global_id(0);

	nImg[id] = lut[A[id]];
	//changes the value of the pixel based on the look up table

	
}