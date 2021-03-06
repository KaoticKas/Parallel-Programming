/*
CMP3752M - Parallel Programming
Kacper Hajda - HAJ17694295
This program uses code adapted from https ://github.com/wing8/OpenCL-Tutorials provided by UOL and the base template was tutorial 2.cpp
The program uses kernels that use atomic operations that were used from tutorial 3 to execute on the input image which can be changed with the -f flag
to firstly transform image into a histogram and then that is transformed into a cumulative histogram using atomic functions as well. This then gets put into a lookup table where it gets normalised and then uses the final kernel to transform the input image into a normalsed image that will be outputted by the end. The execution times and memory transfers are recorded as well as total execution time.
*/

#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"
//importing the required libraries and support files to read the image and import openCL functions
using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
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
		//initalising the binsize to 256 which is an 8 bit image

		//host operations 
		cl::Context context = GetContext(platform_id, device_id);
		//selects the device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/assessment_kernels.cl");
		//adds the kernels that are device code
		cl::Program program(context, sources);

		//build and debug the kernel code, if there is an error throw an error
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
		// creates an int type for the vector to define other vectors
		std::vector<vec_type> H(binSize, 0);
		// creates a vector of type int that is of the predefined binsize
		size_t histo_size = H.size() * sizeof(vec_type);
		//gets the size of the histogram
		std::vector<vec_type> cH(binSize, 0);
		size_t ch_size = cH.size() * sizeof(vec_type);
		// creates a vector for the cumulative histogram and gets the size
		std::vector<vec_type> lut(binSize, 0); // vector for the look up table to be used for the cummulative Histogram

		size_t lut_size = lut.size() * sizeof(vec_type);
		
		
		//sets up buffers

		//device buffers
		cl::Buffer buffer_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		//sets up a buffer for the image
		cl::Buffer buffer_histo_output(context, CL_MEM_READ_WRITE, histo_size);
		//sets up the buffer for the hisogram calculations
		cl::Buffer buffer_histoC_output(context, CL_MEM_READ_WRITE, ch_size); 
		//sets up the buffer for the histogram outputs
		cl::Buffer buffer_LUT_output(context, CL_MEM_READ_WRITE, lut_size);

		cl::Buffer buffer_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		//sets up the output buffer


		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(buffer_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		
		//4.2 Setup and execute the kernel (i.e. device code)
		//histogram kernel to work out the colour histogram for the image
		cl::Kernel histoKernel = cl::Kernel(program, "histo");
		histoKernel.setArg(0, buffer_image_input);
		histoKernel.setArg(1, buffer_histo_output);
		//sets arguments to take in the image and output the vector for the histogram

		cl::Event hisevent;
		// creates an event for the histogram to get the execution and processing time
		queue.enqueueNDRangeKernel(histoKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hisevent);
		queue.enqueueReadBuffer(buffer_histo_output, CL_TRUE, 0, histo_size, &H[0]);
		//adds the kernel to the queue to be executed and reads the result into the histo output buffer into the vector H
		queue.enqueueFillBuffer(buffer_histoC_output, 0, 0, ch_size);
		
		cl::Kernel histoCKernel = cl::Kernel(program, "histoC");
		//sets up the cummulative 
		histoCKernel.setArg(0, buffer_histo_output);
		histoCKernel.setArg(1, buffer_histoC_output);
		histoCKernel.setArg(2, binSize);
		//sets the kernel up with the required variables to work out the cummulative buffer
		cl::Event histoCevent;
		queue.enqueueNDRangeKernel(histoCKernel, cl::NullRange, cl::NDRange(ch_size), cl::NullRange, NULL, &histoCevent);
		queue.enqueueReadBuffer(buffer_histoC_output, CL_TRUE, 0, ch_size, &cH[0]);

		queue.enqueueFillBuffer(buffer_LUT_output, 0, 0, lut_size);

		cl::Kernel lutKernel = cl::Kernel(program, "LUT");
		//sets up the normalised histogram via a look up table 
		lutKernel.setArg(0, buffer_histoC_output);
		lutKernel.setArg(1, buffer_LUT_output);


		cl::Event lutEvent;

		queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(lut_size), cl::NullRange, NULL, &lutEvent);
		queue.enqueueReadBuffer(buffer_LUT_output, CL_TRUE, 0, lut_size, &lut[0]);

		cl::Kernel imgKernel = cl::Kernel(program, "adjustImg");
		//sets up the normalised histogram via a look up table 
		imgKernel.setArg(0, buffer_image_input);
		imgKernel.setArg(1, buffer_LUT_output);
		imgKernel.setArg(2, buffer_image_output);


		cl::Event imgAdjustEvent;

		queue.enqueueNDRangeKernel(imgKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &imgAdjustEvent);
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueReadBuffer(buffer_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		//sets up the image output after adjusting it by the normalised histogram



		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");
		//displays the final normalised image

		std::cout << "Histogram execution time [ns]: " << hisevent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hisevent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Histogram memory transfer:" << GetFullProfilingInfo(hisevent, ProfilingResolution::PROF_US) << std::endl;

		std::cout << "Cumulative Histogram execution time [ns]: " << histoCevent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histoCevent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Cumulative Histogram memory transfer" << GetFullProfilingInfo(histoCevent, ProfilingResolution::PROF_US) << std::endl;

		std::cout << "Lookup table execution time [ns]: " << lutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Lookup table memory transfer:" << GetFullProfilingInfo(histoCevent, ProfilingResolution::PROF_US) << std::endl;

		std::cout << "Image adjust execution time [ns]: " << imgAdjustEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - imgAdjustEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Image adjust memory transfer:" << GetFullProfilingInfo(imgAdjustEvent, ProfilingResolution::PROF_US) << std::endl;

		double totalExecutionTime = (hisevent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hisevent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + histoCevent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histoCevent.getProfilingInfo<CL_PROFILING_COMMAND_START>() + lutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() + imgAdjustEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - imgAdjustEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		std::cout << "total execution time[ns]:" << totalExecutionTime<< std::endl;

		//gets the execution times of kernels


 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }
		//exit program

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
