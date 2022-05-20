#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImg<unsigned char> input_image_8;
		CImgDisplay disp_input(image_input,"input");
		int binSize = 256;


		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//part 4 allocate memory to the histogram and cummulative Histogram
		typedef int vec_type;
		// creates an int type for the vector
		std::vector<vec_type> H(binSize, 0);
		// creates a vector of type int that is of the predefined binsize
		size_t histo_size = H.size() * sizeof(vec_type);

		std::vector<vec_type> cH(binSize, 0);
		size_t ch_size = cH.size() * sizeof(vec_type);
		// creates a vector for the cumulative histogram and gets the size
		std::vector<vec_type> LUT(binSize, 0); // vector for the look up table to be used for the cummulative Histogram

		size_t LUT_size = LUT.size() * sizeof(vec_type);
		//Part 6 

		//device buffers
		cl::Buffer buffer_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		//sets up a buffer for the image
		cl::Buffer buffer_histo_output(context, CL_MEM_READ_WRITE, histo_size);
		//sets up the buffer for the hisogram calculations
		cl::Buffer buffer_histoC_output(context, CL_MEM_READ_WRITE, ch_size); 
		//sets up the buffer for the histogram outputs
		cl::Buffer buffer_LUT_output(context, CL_MEM_READ_WRITE, LUT_size);

		cl::Buffer buffer_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		//sets up the output buffer


		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(buffer_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		
		//4.2 Setup and execute the kernel (i.e. device code)
		//histogram kernel to work out the colour histogram for the image
		cl::Kernel histoKernel = cl::Kernel(program, "histo");
		histoKernel.setArg(0, buffer_image_input);
		histoKernel.setArg(1, buffer_histo_output);

		cl::Event hisevent;// creates an event for the histogram to get the execution and processing time
		queue.enqueueNDRangeKernel(histoKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hisevent);
		queue.enqueueReadBuffer(buffer_histo_output, CL_TRUE, 0, histo_size, &H[0]);
		
		queue.enqueueFillBuffer(buffer_histoC_output, 0, 0, ch_size);

		cl::Kernel histoCKernel = cl::Kernel(program, "histoC");
		//sets up the cummulative 
		histoCKernel.setArg(0, buffer_histo_output);
		histoCKernel.setArg(1, buffer_histoC_output);

		cl::Event histoCevent;
		queue.enqueueNDRangeKernel(histoCKernel, cl::NullRange, cl::NDRange(ch_size), cl::NullRange, NULL, &histoCevent);
		queue.enqueueReadBuffer(buffer_histoC_output, CL_TRUE, 0, ch_size, &cH[0]);

		queue.enqueueFillBuffer(buffer_histoC_output, 0, 0, ch_size);









		vector<unsigned char> output_buffer(image_input.size());



		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());

		CImgDisplay disp_output(output_image,"output");


 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
