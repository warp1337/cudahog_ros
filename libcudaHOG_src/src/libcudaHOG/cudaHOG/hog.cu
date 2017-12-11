//
//    Copyright (c) 2009-2011
//      Patrick Sudowe	<sudowe@umic.rwth-aachen.de>
//      RWTH Aachen University, Germany
//
//    This file is part of groundHOG.
//
//    GroundHOG is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GroundHOG is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with groundHOG.  If not, see <http://www.gnu.org/licenses/>.
//


#include <cuda.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <vector>

#include "global.h"
#include "gradients.h"
#include "padding.h"
#include "conversions.h"
#include "blocks.h"
#include "svm.h"
#include "descriptor.h"
#include "detections.h"
#include "nms.h"
#include "timer.h"
#include "roi.h"
#include "parameters.h"

#include "cudaHOG.h"

namespace cudaHOG {

double total_gradients, total_blocks, total_svm, total_end, total_nms;

float* d_pBlocks = NULL;
float2* d_pGradMag = NULL;

int hog_initialize()
{
// ------------------------------------------------------------------------
//	check cuda device

	int deviceCount = 0;
	if( cudaGetDeviceCount( &deviceCount) ) {
		printf("cudaGetDeviceCount failed\n");
		printf("CUDA driver and runtime version may be mismatched!\n");
	}

	if( deviceCount == 0 ) {
		printf("sorry no CUDA capable device found.\n");
		return -1;
	}

	int dev;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	}

	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("driver: %d\nruntime: %d\n", driverVersion / 1000, runtimeVersion / 1000);


// ------------------------------------------------------------------------
//	prepare weights
	prepareGaussWeights();
	prepareBilinearWeights();
// ------------------------------------------------------------------------
//	malloc all memory that will be needed during processing
	// gradients
	cudaMalloc((void**)&d_pGradMag, MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION * sizeof(float2));
		ONFAIL("d_pGradMag malloc");
	cudaMemset(d_pGradMag, 0, sizeof(float2) * MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION);
		ONFAIL("d_pGradMag memset");

	// blocks
	const int nBlocks = MAX_IMAGE_DIMENSION/8 * MAX_IMAGE_DIMENSION/8 ;	// WE ASSUME MAXIMUM IMAGE SIZE OF 1280x1280
	const int blocksMemorySize = nBlocks * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS * sizeof(float);
	cudaMalloc((void**)&d_pBlocks, blocksMemorySize);
		ONFAIL("cudaMalloc for block histograms failed")

	return 0;
}

int hog_finalize()
{
	cudaFree(d_pGradMag); d_pGradMag = NULL;
		ONFAIL("cudaFree: d_pGradMag");
	cudaFree(d_pBlocks); d_pBlocks = NULL;
		ONFAIL("cudaFree: d_pBlocks");
	if( svm_finalize() || blocks_finalize() )
		return -1;
	return 0;
}

int hog_transfer_image(const unsigned char* h_pImg, int width, int height)
{
// preprocess image (bind to texture memory)
//	printf("img dim: %d x %d\n", width, height);

	if( prepare_image(h_pImg, width, height) ) {
		printf("prepare_image failed\n");
		return -1;
	}

	return 0;
}

int hog_release_image()
{
	return destroy_image();
}


int hog_process_image(int width, int height, float scale,
						int padX, int padY, ROI* roi, int* cntBlocks, int* cntSVM,
						MultiDetectionList& multi_detections)
{
	Timer gradients, blocks, svm, end;

	int res;
// ------------------------------------------------------------------------
//	compute gradients
	startTimer(&gradients);

	// pad the original image
	if(padX == -1 || padY == -1) {
		padX = HOG_PADDING_X;
		padY = HOG_PADDING_Y;
	}

	// size of padded image
	int paddedWidth = padX * 2 + width;
	int paddedHeight= padY * 2 + height;
	// region of interest - do computation only for this region
	int min_x = 0;
	int min_y = 0;
	int max_x = width;
	int max_y = height;
	if(roi) {
		min_x = max(0, roi->min_x -1);
		min_y = max(0, roi->min_y -1);
		max_x = min(paddedWidth, roi->max_x);
		max_y = min(paddedHeight, roi->max_y);
	}

	// size of the resulting gradient image
	int gradWidth = (max_x-min_x) +2*padX -1;
	int gradHeight = (max_y-min_y)+2*padY -1;

if( PRINT_DEBUG_INFO ) {
	printf("scale: %f\t\tw x h: %d\t\t%d\t\t%d\t\t",
			scale, gradWidth, gradHeight, gradHeight * gradWidth);
}

	cudaMemset(d_pGradMag, 0, sizeof(float2) * gradHeight * gradWidth);
		ONFAIL("d_pGradMag memset");

	res = compute_gradients(paddedWidth, paddedHeight, min_x, min_y, max_x, max_y, padX, padY, d_pGradMag);
	if( res ) {
		printf("compute_gradients failed: %d\n", res);
		return -3;
	}

	stopTimer(&gradients);
// ------------------------------------------------------------------------
//	compute blocks
	startTimer(&blocks);

	dim3 blockgrid;
	blockgrid.x = (int)floor(gradWidth / 8.f);
	blockgrid.y = (int)floor(gradHeight / 8.f);
	res = compute_blocks(blockgrid, gradWidth, gradHeight, d_pGradMag, d_pBlocks);
	if( res ) {
		printf("compute_blocks failed: %d\n", res);
		return -4;
	}

	*cntBlocks += blockgrid.x * blockgrid.y;
if( PRINT_DEBUG_INFO ) {
	printf("%d\n", blockgrid.x * blockgrid.y);
}

	stopTimer(&blocks);
// ------------------------------------------------------------------------
//	compute descriptors & evaluate blocks with SVM
	startTimer(&svm);

	int cnt;
	res = svm_evaluate(d_pBlocks, blockgrid.x, blockgrid.y,
						padX, padY, min_x, min_y, scale,
						&cnt, multi_detections);
	if( res ) {
		printf("svm_evaluate failed: %d\n", res);
		return -5;
	}
	*cntSVM += cnt;

	stopTimer(&svm);
// ------------------------------------------------------------------------
//	check for positive detections - and save to result datastructure
	startTimer(&end);

	stopTimer(&end);

	total_gradients += getTimerValue(&gradients);
	total_blocks += getTimerValue(&blocks);
	total_svm += getTimerValue(&svm);
	total_end += getTimerValue(&end);

//	printf("gradients: %.3f\n", getTimerValue(&gradients));
//	printf("blocks: %.3f\n", getTimerValue(&blocks));
//	printf("svm: %.3f\n", getTimerValue(&svm));
//	printf("end: %.3f\n", getTimerValue(&end));

	return 0;
}


int hog_process_image_multiscale(int width, int height, std::vector<ROI>& roi, int* cntBlocks, int* cntSVM,
							double* timings, MultiDetectionList& detections)
{
	Timer global_timer;
	startTimer(&global_timer);

	total_gradients=  0.; total_blocks = 0.; total_svm = 0.; total_end = 0.;

	MultiDetectionList detections_all;

	float startScale = HOG_START_SCALE;
	int min_window_width = g_params.min_window_width();
	int min_window_height = g_params.min_window_height();

	float endScale = min( ( width + 2*HOG_PADDING_X ) / (float)min_window_width ,
		  					(height+ 2*HOG_PADDING_Y ) / (float)min_window_height );

if( PRINT_DEBUG_INFO ) {
	printf("min_window_width: %d\t min_window_height: %d\n", min_window_width, min_window_height);
	printf("endScale: %f\n", endScale);
}

	std::vector<float> scales;
        if( roi.size() == 0 ) {
		float scale = startScale;
		size_t i=0;
		while( scale < endScale ) {
                        scales.push_back(scale);
			scale *= (float)HOG_SCALE_STEP;
			i++;
		}
	} else {
		scales.resize(roi.size());
		for(size_t i=0; i < roi.size(); i++)
			scales[i] = roi[i].scale;
	}

	int count = 0;
	for(; count < scales.size(); count++) {
		const float scale = scales[count];

		int curwidth = width / scale;
		int curheight = height / scale;

		assert(curwidth < MAX_IMAGE_DIMENSION );
		assert(curheight < MAX_IMAGE_DIMENSION );

if( DEBUG_PRINT_PROGRESS ) {
		printf("w x h: %d\t%d\t\t\t", curwidth, curheight);
		printf("current scale: %f\n", scale);
}

		if(roi.size() > 0 ) {
			assert(count < roi.size() );

			if( roi[count].min_x == INT_MAX ) {
				if(DEBUG_PRINT_SCALE_CONSTRAINTS)
					printf("invalid ROI - skipping to next scale\n");
				continue;
			}
			if(DEBUG_PRINT_SCALE_CONSTRAINTS)
				printf("processing at %f\n", scale);

			// no padding when using a ROI constraint
			hog_process_image(curwidth, curheight, scale,
						0, 0, &(roi[count]), cntBlocks, cntSVM, detections_all);
		} else {
			int oneValidModel = 0;
			int ii;
			for(ii = 0; ii < g_params.models.size(); ii++ ) {
				if(    scale >= g_params.models[ii].min_scale
					|| scale <= g_params.models[ii].max_scale )
					oneValidModel = 1;
			}
			if( !oneValidModel ) {
				if(DEBUG_PRINT_SCALE_CONSTRAINTS)
					printf("hog_process_image_multiscale: skipping scale: %f \t no model to be evaluted here\n", scale);
				continue;
			}

			hog_process_image(curwidth, curheight, scale,
						HOG_PADDING_X, HOG_PADDING_Y, NULL, cntBlocks, cntSVM, detections_all);
		}
	}

if( PRINT_VERBOSE_INFO ) {
	int sum = 0;
	for(size_t i=0; i < detections_all.size(); i++)
		sum += detections_all[i].size();
	printf("processing %d detections\n", sum);
}

// perform Non-Maximum-Suppression on the initial detections
	Timer nms_timer;
	startTimer(&nms_timer);

	detections.resize(detections_all.size());
if( SKIP_NON_MAXIMUM_SUPPRESSION ) {
	std::copy(detections_all.begin(), detections_all.end(), detections.begin());
} else {
	for(int ii=0; ii < detections_all.size(); ii++) {
		nms_process_detections((detections_all[ii]), (detections[ii]), g_params.models[ii]);
	}
}

	stopTimer(&nms_timer);

	stopTimer(&global_timer);

	total_nms = getTimerValue(&nms_timer);

if( PRINT_PROFILING_TIMINGS ) {
	printf("\nevaluated %d scales\n", count);
	printf("processing time: %.2f ms\n", getTimerValue(&global_timer));

	printf("\ntotal processing time:\n");
	double total = total_gradients + total_blocks + total_svm + total_end + total_nms;
	printf("gradients:\t%f ms \t(%.2f%%)\n", total_gradients, total_gradients / total *100.);
	printf("blocks:\t\t%f ms \t(%.2f%%)\n", total_blocks, total_blocks / total * 100.);
	printf("svm:\t\t%f ms \t(%.2f%%)\n", total_svm, total_svm / total * 100.);
	printf("end:\t\t%f ms \t(%.2f%%)\n", total_end, total_end / total * 100.);
	printf("nms:\t\t%.2f ms \t(%.2f%%)\n", getTimerValue(&nms_timer), total_nms / total * 100. );
}

// add overall timings for this frame
	timings[0] += total_gradients;
	timings[1] += total_blocks;
	timings[2] += total_svm;
	timings[3] += total_nms;
	timings[4] += getTimerValue(&global_timer);

	detections_all.clear();
	return 0;
}


int hog_get_descriptor(int width, int height, int bPad,
						int featureX, int featureY, float scale,
						ModelParameters& params,
						float* h_pDescriptor)
{
	if(!h_pDescriptor) return -10;

	int padX = 0;
	int padY = 0;
	if( bPad ) {
		padX = HOG_PADDING_X;
		padY = HOG_PADDING_Y;
	}

	int w = (int)(width/scale);
	int h = (int)(height/scale);

	int paddedWidth = padX * 2 + (int)(width / scale);
	int paddedHeight= padY * 2 + (int)(height / scale);

	int res = compute_gradients(paddedWidth, paddedHeight, 0, 0, w, h, padX, padY, d_pGradMag);
	if( res ) {
		printf("w x h : %d x %d \t s: %.4f\n", paddedWidth, paddedHeight, scale);
		printf("compute_gradients failed: %d\n", res);
		return -3;
	}

	// alloc and initialize memory for output
	dim3 grid;
	grid.x = (int)floorf((paddedWidth-1) / 8);
	grid.y = (int)floorf((paddedHeight-1) / 8);
	// the gradient image is 1 pixel smaller (therefore -1)
	compute_blocks(grid, paddedWidth-1, paddedHeight-1, d_pGradMag, d_pBlocks);
		ONFAIL("compute_blocks kernel failed");

	float* d_pDescriptor;
	cudaMalloc((void**)&d_pDescriptor, sizeof(float) * params.dimension());
	cudaMemset(d_pDescriptor, 0, sizeof(float) * params.dimension());

	compute_descriptor(d_pDescriptor, d_pBlocks, featureX, featureY, grid.x, params);

	cudaMemcpy(h_pDescriptor, d_pDescriptor, sizeof(float) * params.dimension(),
				cudaMemcpyDeviceToHost);
	cudaFree(d_pDescriptor);
	return 0;
}

}	// end of namespace cudaHOG
