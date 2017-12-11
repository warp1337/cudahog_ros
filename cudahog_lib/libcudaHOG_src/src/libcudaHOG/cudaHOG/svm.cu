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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

#include "global.h"
#include "parameters.h"
#include "detections.h"

#ifdef WIN32
typedef __int64 int64_t;
#endif

texture<float, 1, cudaReadModeElementType> t_weights;

float* d_pResults = NULL;
float* h_pResults = NULL;

namespace cudaHOG {

class Model {
public:
	Model()
		: nWeights(0), bias(0), h_pWeights(NULL), d_pWeights(NULL) {};
	long nWeights;
	double bias;
	float* h_pWeights;
	float* d_pWeights;
};

std::vector<Model> _models;


// function reads the binary SVM model format of SVMdense
// there are various versions of this formats, which are handled accordingly
int read_binary_model(const char* fnSVMModel, Model& model)
{
  FILE *fp;

  int verbosity = 0;

  if(verbosity>=2) {
    printf("Reading model..."); fflush(stdout);
  }
  if ((fp = fopen (fnSVMModel, "rb")) == NULL)
  { printf ("failed to open model file: %s\n", fnSVMModel); return -1; }

  char version_buffer[10];
  if (!fread (&version_buffer,sizeof(char),10,fp))
  { printf ("Unable to read version"); return -1; }
  /* read version number */
  int version = 0;
  if (!fread (&version,sizeof(int),1,fp))
  { printf ("Unable to read version number"); return -1; }

#ifdef PRINT_VERBOSE_INFO
  printf("model file version: %d\n", version);
#endif
  if (version < 200)
  { printf("Model file compiled for light version"); return -1; }

  fseek(fp, sizeof(int64_t) * 2  + sizeof(double) * 3, SEEK_CUR);
  int64_t l;
  fread(&l,sizeof(int64_t),1,fp);
  fseek(fp, l*sizeof(char), SEEK_CUR);
  fread(&(model.nWeights),sizeof(int64_t),1,fp);
  fseek(fp, sizeof(int64_t), SEEK_CUR);

  if(version >= 201) {
    fseek(fp, 2*sizeof(double), SEEK_CUR);
  }

  fseek(fp, sizeof(int64_t), SEEK_CUR);
  fread(&(model.bias), sizeof(double),1,fp);

	model.h_pWeights =(float*)malloc(sizeof(float)*(model.nWeights+1));
	assert(model.h_pWeights  != NULL);

	double* tmpweights  = (double*)malloc(sizeof(double)*(model.nWeights+1));
	assert(tmpweights  != NULL);

	int cnt = fread(tmpweights, sizeof(double),model.nWeights+1,fp);
	double* tmpptr = tmpweights;
	float* mptr = model.h_pWeights;
	for(int j=0; j < model.nWeights+1; j++, tmpptr++, mptr++) {
		*(mptr) = (float)(*tmpptr);
	}
	free(tmpweights);

	fclose(fp);

	return 0;
}


int svm_initialize()
{
	// prepare malloc with maximum size ever needed
	cudaMalloc((void**)&d_pResults,
							sizeof(float) *
							(MAX_IMAGE_DIMENSION/8) *
							(MAX_IMAGE_DIMENSION/8));
		ONFAIL("svm_initialize: cudaMalloc")

	cudaMemset(d_pResults, 0, sizeof(float) *
							(MAX_IMAGE_DIMENSION/8) *
							(MAX_IMAGE_DIMENSION/8));

	h_pResults = (float*)malloc(sizeof(float)* MAX_IMAGE_DIMENSION/8 * MAX_IMAGE_DIMENSION/8);

	return 0;
}


int svm_finalize()
{
	// free all models
	for(size_t ii=0; ii < _models.size(); ii++) {
		if(!_models[ii].h_pWeights)
			continue;

		free(_models[ii].h_pWeights);
		cudaFree(_models[ii].d_pWeights);
	}
	_models.clear();

	free(h_pResults);
	return 0;
}

int svm_add_model(const char* fnModel)
{
	Model model;

	if(!fnModel) {	// no file given - do not initialize a model
		model.h_pWeights = NULL;
		model.nWeights = NULL;
		model.bias = 0;
	} else {
#ifdef PRINT_VERBOSE_INFO
		printf("reading SVM model from file: %s\n", fnModel);
#endif
		if( read_binary_model(fnModel, model) )
		{
			printf("read_binary_model failed\n");
			return -1;
		}

	}

	cudaMalloc((void**)&(model.d_pWeights), sizeof(float) * model.nWeights);
		ONFAIL("cudaMalloc for d_pWeights")
	cudaMemset(model.d_pWeights, 0, sizeof(float) * model.nWeights);
	cudaMemcpy(model.d_pWeights, model.h_pWeights, sizeof(float) * model.nWeights, cudaMemcpyHostToDevice);
		ONFAIL("cudaMemcpy: failed to copy SVM weights to device")

	_models.push_back(model);
	return 0;
}

// Use 256 threads to compute SVM
// Only 252 values are computed, so 4 threads idle
// This is a good trade-off, therefore 7 HOG blocks are computed with one thread block
// here. Fortunately, for pedestrians this corresponds to one 'row' of blocks within
// a descriptor window.
__global__ void d_svm_seven(float* d_pBlocks, float bias, int nBlockX, float* d_pResult)
{
	const int idx = threadIdx.x;

	volatile __shared__ float s_results[256];
	s_results[idx] = 0.f;

	// figure out x y position of this descriptor (within the grid of blocks!)
	const int dx = blockIdx.x;
	const int dy = blockIdx.y;

	const int floatsPerDescriptorRow = 7 * 4 * 9;	// 7 blocks * 4 cells * 9 bins

	if( idx < 252 ) {
	for(int row=0; row < 15; row++) {
		const int weights_index = row * floatsPerDescriptorRow;
		float* block = d_pBlocks + 36 * ((dy+row) * nBlockX + dx);
		s_results[idx] += tex1Dfetch(t_weights, weights_index + idx) * block[idx];
	}
	}
	__syncthreads();

	if( idx < 128 ) {
		s_results[idx] += s_results[idx + 128];
	}
	__syncthreads();
	if( idx < 64 ) {
		s_results[idx] += s_results[idx + 64];
	}
	__syncthreads();
	if( idx < 32 ) {
		s_results[idx] += s_results[idx + 32];
		s_results[idx] += s_results[idx + 16];
		s_results[idx] += s_results[idx + 8];
		s_results[idx] += s_results[idx + 4];
		s_results[idx] += s_results[idx + 2];
		s_results[idx] += s_results[idx + 1];
	}
	if( idx == 0 ) {
		d_pResult[blockIdx.y * gridDim.x + blockIdx.x] = s_results[0] - bias;
	}
}

__global__ void d_svm_standard(float* d_pBlocks, float bias,
						 int nBlockX, int descriptor_width, int descriptor_height,
						float* d_pResult)
{
	const int idx = threadIdx.x;

	volatile __shared__ float s_results[128];
	s_results[idx] = 0.f;

	// figure out x y position of this descriptor (within the grid of blocks!)
	const int dx = blockIdx.x;
	const int dy = blockIdx.y;

	const int floatsPerDescriptorRow = descriptor_width * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS;
	const int floatsPerBlock = HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS;

	const int offset = 9 * idx;
	if( offset < floatsPerDescriptorRow ) {

		for(int row=0; row < descriptor_height; row++) {

			const int weights_index = row * floatsPerDescriptorRow;
			float* block = d_pBlocks + floatsPerBlock * ((dy+row) * nBlockX + dx);

			for(int i=0; i < 9; i++) {
				s_results[idx] += tex1Dfetch(t_weights, weights_index + offset + i) * block[offset+i];
			}
		}
	}
	__syncthreads();

	if( idx < 64 ) {
		s_results[idx] += s_results[idx + 64];
	}
	__syncthreads();
	if( idx < 32 ) {
		s_results[idx] += s_results[idx + 32];
		s_results[idx] += s_results[idx + 16];
		s_results[idx] += s_results[idx + 8];
		s_results[idx] += s_results[idx + 4];
		s_results[idx] += s_results[idx + 2];
		s_results[idx] += s_results[idx + 1];
	}
	if( idx == 0 ) {
		d_pResult[blockIdx.y * gridDim.x + blockIdx.x] = s_results[0] - bias;
	}
}

// evaluate all possible descriptor windows directly on the GPU
// without copying&creating individual descriptors first.
// The results are copied to host memory for further processing
int svm_evaluate_one_model(int modelIdx, float* d_pBlocks, int nBlockX, int nBlockY, int* cntSVM, float* h_pResults)
{
	const int descriptor_width = g_params.models[modelIdx].HOG_DESCRIPTOR_WIDTH;
	const int descriptor_height = g_params.models[modelIdx].HOG_DESCRIPTOR_HEIGHT;
	const int nDescriptorX = nBlockX - descriptor_width +1;
	const int nDescriptorY = nBlockY - descriptor_height +1;

	// bind the correct weights to the texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture(0, t_weights, _models[modelIdx].d_pWeights, channelDesc,
					sizeof(float) * _models[modelIdx].nWeights);
		ONFAIL("cudaBindTexture")

	memset(h_pResults, 0, sizeof(float) * nDescriptorX * nDescriptorY);

	if( 	g_params.models[modelIdx].HOG_DESCRIPTOR_WIDTH == 7
		&&	g_params.models[modelIdx].HOG_DESCRIPTOR_HEIGHT == 15 ) {
		// specially optimized routine for pedestrian descriptors

		dim3 grid(nDescriptorX, nDescriptorY, 1);	// one block for each descriptor
		dim3 threads(256, 1, 1); // 256 threads compute one hog block row of the descriptor (252 values)
		d_svm_seven<<< grid, threads>>>(d_pBlocks, _models[modelIdx].bias, nBlockX, d_pResults);

	} else {
		const int nThreads = 128;
		if( descriptor_width * 4 > nThreads ) {
			printf("WARNING: SVM evaluation routine will not work\n");
			return -1;
		}

		dim3 grid(nDescriptorX, nDescriptorY, 1);	// one block for each descriptor
		// 128 threads compute one hog block row of the descriptor
		dim3 threads(nThreads, 1, 1);
		d_svm_standard<<<grid, threads>>>(d_pBlocks, _models[modelIdx].bias, nBlockX,
										descriptor_width, descriptor_height, d_pResults);
	}

	*cntSVM = nDescriptorX * nDescriptorY;

	cudaMemcpy(h_pResults, d_pResults, sizeof(float) * nDescriptorX * nDescriptorY, cudaMemcpyDeviceToHost);
		ONFAIL("svm_evaluate_direct: memcpy")

	cudaUnbindTexture(t_weights);
		ONFAIL("svm_evaluate_direct: unbind texture")
	return 0;
}

int svm_evaluate(float* d_pBlocks,
		int nBlockX, int nBlockY,
		int padX, int padY, int minX, int minY, float scale,
		int* cntSVM,
		MultiDetectionList& detections)
{
	assert( _models.size() > 0 );

	detections.resize(_models.size());
	for(size_t ii=0; ii < _models.size(); ii++) {
		const int w = g_params.models[ii].HOG_DESCRIPTOR_WIDTH;
		const int h = g_params.models[ii].HOG_DESCRIPTOR_HEIGHT;

		if( nBlockX <= w || nBlockY <= h ) {
			if(DEBUG_PRINT_SCALE_CONSTRAINTS)
				printf("svm_evaluate: Skipping model %s at scale %f -- ROI too small\n", g_params.models[ii].identifier.c_str(), scale);
			continue;
		}

		// if we get here, at least one model has to be evaluated, but not necessarily all
		// So we check again here. This check is performed both for detection with and without groundplane
		if(    scale > g_params.models[ii].max_scale
			|| scale < g_params.models[ii].min_scale ) {
			if(DEBUG_PRINT_SCALE_CONSTRAINTS)
				printf("svm_evaluate: Skipping model %d at scale %f -- due to scale range from config file\n", ii, scale);
			continue;
		}
		else {
 if( PRINT_DEBUG_INFO ) {
			printf("\nEvaluating model %s at scale %f\n\n", g_params.models[ii].identifier.c_str(), scale);
 }
		}

	// evaluate SVM
		if( svm_evaluate_one_model(ii, d_pBlocks, nBlockX, nBlockY, cntSVM, h_pResults) )
			return -1;

	// copy results to host mem & save positve results
		const int nDescriptorX = nBlockX - w +1;
		const int nDescriptorY = nBlockY - h +1;

		for(int y=0; y < nDescriptorY; y++) {
			for(int x=0; x < nDescriptorX; x++) {
				float score = h_pResults[y*nDescriptorX + x];
				if( score > 0.f ) {
					int posx = (x * HOG_CELL_SIZE -padX + minX) * scale ;
					int posy = (y * HOG_CELL_SIZE -padY + minY) * scale ;

					detections[ii].push_back(Detection(posx, posy, scale, score));
				}
			}
	}

	}
	return 0;
}

} // end of namespace cudaHOG
