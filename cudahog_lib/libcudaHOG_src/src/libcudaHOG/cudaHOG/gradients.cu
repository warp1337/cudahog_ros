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
#include <time.h>

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "global.h"

static cudaArray* normalized_image_array = NULL;
texture<uchar4, 2, cudaReadModeNormalizedFloat> t_normalized_image_texture;

// compute the image gradients & magnitude for each pixel
__global__ void d_compute_gradients(int width, int height,
								int min_x, int min_y, int max_x, int max_y,
								int padX, int padY, float2* d_pGradMag)
{
	const float xstep = 1.f / (width-2*padX);
	const float ystep = 1.f / (height-2*padY);
	const float xoff = xstep / 2.f;
	const float yoff = ystep / 2.f;

	float4	pixel_up;
	float4	pixel_down;
	float4	pixel_left;
	float4	pixel_right;

	const int posx	= blockDim.x * blockIdx.x + threadIdx.x + min_x;// pixel pos within padded image
	const int posy	= blockDim.y * blockIdx.y + threadIdx.y + min_y;// pixel pos within padded image


	// pixel pos in gradmag 'image'
	// the gradient image is 1 pixels smaller in both dimensions
	const int roi_width  = max_x - min_x + 2*padX-1;
//	const int pixelIdx = (posx-min_x-1) + (posy-min_y-1) * roi_width;
	const int pixelIdx = (posx-min_x) + (posy-min_y) * roi_width;

	// initialize border (no gradient information for border pixels)
	if(	posx == 0 || posy == 0 || posx == max_x || posy == max_y ) {
		d_pGradMag[pixelIdx].x = 0.f;
		d_pGradMag[pixelIdx].y = 0.f;
	}

	if(posx < max_x+2*padY && posx > 0 && posy < max_y+2*padY && posy > 0) {
		// the indizes can be <0 and >1 ! this implicitely pads the image
		pixel_down = tex2D(t_normalized_image_texture,
								(posx-padX) * xstep + xoff,
								(posy+1-padY) * ystep + yoff);
		pixel_up = tex2D(t_normalized_image_texture,
								(posx-padX) * xstep + xoff,
								(posy-1-padY) * ystep + yoff);
		pixel_left = tex2D(t_normalized_image_texture,
								(posx-1-padX) * xstep + xoff,
								(posy-padY) * ystep + yoff);
		pixel_right = tex2D(t_normalized_image_texture,
								(posx+1-padX) * xstep + xoff,
								(posy-padY) * ystep + yoff);

#ifdef ENABLE_GAMMA_COMPRESSION
		pixel_up.x = sqrtf(	pixel_up.x);
		pixel_up.y = sqrtf(	pixel_up.y);
		pixel_up.z = sqrtf(	pixel_up.z);
		pixel_up.w = sqrtf(	pixel_up.w);
		pixel_down.x = sqrtf(pixel_down.x);
		pixel_down.y = sqrtf(pixel_down.y);
		pixel_down.z = sqrtf(pixel_down.z);
		pixel_down.w = sqrtf(pixel_down.w);
		pixel_left.x = sqrtf(pixel_left.x);
		pixel_left.y = sqrtf(pixel_left.y);
		pixel_left.z = sqrtf(pixel_left.z);
		pixel_left.w = sqrtf(pixel_left.w);
		pixel_right.x = sqrtf(pixel_right.x);
		pixel_right.y = sqrtf(pixel_right.y);
		pixel_right.z = sqrtf(pixel_right.z);
		pixel_right.w = sqrtf(pixel_right.w);
#endif
		// compute gradient direction and magnitude
		float3 grad_dx, grad_dy;
		grad_dx.x = (pixel_right.x - pixel_left.x);
		grad_dx.y = (pixel_right.y - pixel_left.y);
		grad_dx.z = (pixel_right.z - pixel_left.z);

		grad_dy.x = (pixel_down.x - pixel_up.x);
		grad_dy.y = (pixel_down.y - pixel_up.y);
		grad_dy.z = (pixel_down.z - pixel_up.z);

		float3 mag;
		mag.x = grad_dx.x * grad_dx.x + grad_dy.x * grad_dy.x;
		mag.y = grad_dx.y * grad_dx.y + grad_dy.y * grad_dy.y;
		mag.z = grad_dx.z * grad_dx.z + grad_dy.z * grad_dy.z;

		float direction;
		float magnitude;
		if(mag.z > mag.y) {
			if(mag.z > mag.x) {
				// z - red
				magnitude	= sqrtf(mag.z);
				direction	= atan2f(grad_dy.z, grad_dx.z) * 180.f / (float)M_PI;
			} else {
				// x - blue
				magnitude	= sqrtf(mag.x);
				direction	= atan2f(grad_dy.x, grad_dx.x) * 180.f / (float)M_PI;
			}
		} else {
			if(mag.y > mag.x) {
				// y - green
				magnitude	= sqrtf(mag.y);
				direction	= atan2f(grad_dy.y, grad_dx.y) * 180.f / (float)M_PI;
			} else {
				// x - blue
				magnitude	= sqrtf(mag.x);
				direction	= atan2f(grad_dy.x, grad_dx.x) * 180.f / (float)M_PI;
			}
		}

		d_pGradMag[pixelIdx].x = direction;
		d_pGradMag[pixelIdx].y = magnitude;
	}
}


int prepare_image(const unsigned char* h_pImg, int width, int height)
{
	// allocate array - copy image there - bind it to texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray( &normalized_image_array, &channelDesc, width, height);
		ONFAIL("malloc array\n");

	cudaMemcpyToArray(normalized_image_array, 0, 0, h_pImg, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
		ONFAIL("memcpy to array\n");

	t_normalized_image_texture.addressMode[0] = cudaAddressModeClamp;
	t_normalized_image_texture.addressMode[1] = cudaAddressModeClamp;
	t_normalized_image_texture.filterMode = cudaFilterModeLinear;
	t_normalized_image_texture.normalized = true;

	cudaBindTextureToArray( t_normalized_image_texture, normalized_image_array, channelDesc);
		ONFAIL("bind tex to array\n");

	return 0;
}


int destroy_image()
{
	if( normalized_image_array ) {
		cudaUnbindTexture( t_normalized_image_texture );
			ONFAIL("failed to unbind texture")
		cudaFreeArray( normalized_image_array );
			ONFAIL("cudaFreeArray failed")
		normalized_image_array = NULL;
	}
	return 0;
}


__global__ void d_rescale_image(float4* d_pImg, int width, int height, int padX, int padY)
{
	// we do this slow and save
	if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) {

		// add 'virtual' padding of 16 extra pixels in each direction
		float xstep = 1.f / (width-2*padX);
		float ystep = 1.f / (height-2*padY);

		float xpos = -padX * xstep + xstep/2.f;
		float ypos = -padY * ystep + ystep/2.f;

		for(int y=0; y < height; y++, ypos += ystep ) {
			for(int x=0; x < width; x++, xpos += xstep ) {
				*(d_pImg + width*y +x)  = tex2D(t_normalized_image_texture, xpos, ypos);
			}
			xpos = -padX * xstep + xstep/2.f;
		}

	}
}


int test_prepared_image(float scale, int origwidth, int origheight, int padX, int padY)
{
	// read the image at a certain scale
	printf("pad: %d x %d\n", padX, padY);
	printf("orig dim: %d x %d\n", origwidth, origheight);

	int width = origwidth * scale + 2*padX;
	int height = origheight * scale + 2*padY;

	printf("test_rescale_image:\n");
	printf("image dimensions after rescaling: %d x %d\n", width, height);

	// alloc device mem for rescaled image
	float4* d_pRescaled;
	cudaMalloc((void**)&d_pRescaled, sizeof(float4)*width*height);

	// rescale the image on device (using image texture)
	dim3 g(8,1,1);
	dim3 t(8,1,1);
	d_rescale_image<<<g,t>>> (d_pRescaled, width, height, padX, padY);

	// copy the image to host mem
	float4* h_pRescaled = (float4*)malloc(sizeof(float4) * width * height);
	cudaMemcpy(h_pRescaled, d_pRescaled, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);

	for(int i=0; i < 3* width; i++) {
		float4 data = *(h_pRescaled+i);
		printf("(%f,%f,%f,%f)\n", data.x, data.y, data.z, data.w);
	}
	printf("\n");

	// save to file
    FILE *fpb;
    fpb = fopen("rescaled_image.ppm", "wb" );
    fprintf( fpb, "P6\n%d %d\n255\n", width, height );
    for( int iy = 0; iy < height; iy++) {
        for( int ix = 0; ix < width; ix++ ) {
			float4 data = *(h_pRescaled + iy * width + ix);
			unsigned char r = (unsigned char)(data.z * 255);
			unsigned char g = (unsigned char)(data.y * 255);
			unsigned char b = (unsigned char)(data.x * 255);
			//unsigned char a = (unsigned char)(data.w * 255);
			fwrite(&r,1,1,fpb);
			fwrite(&g,1,1,fpb);
			fwrite(&b,1,1,fpb);
		}
	}
    fclose( fpb );

	printf("success.\n");
	return 0;
}


__host__ int compute_gradients( int paddedWidth, int paddedHeight,
								int min_x, int min_y, int max_x, int max_y,
								int padX, int padY, float2* d_pGradMag)
{
// compute the gradients

	// start kernel to compute gradient directions & magnitudes
	const int TX = 16;
	const int TY = 16;
	dim3 threads(TX, TY);
	dim3 grid( (int)ceil(paddedWidth/((float)TX)), (int)ceil(paddedHeight/((float)TY)) );

//	printf("\ncompute_gradients:\n");
//	printf("img dim: %d x %d\n", paddedWidth, paddedHeight);
//	printf("grid: %d, %d, %d\n", grid.x, grid.y, grid.z);

	d_compute_gradients<<< grid , threads >>>(paddedWidth, paddedHeight,
												min_x, min_y, max_x, max_y,
												padX, padY, d_pGradMag);
		ONFAIL("compute_gradients\n");

#ifdef DEBUG_DUMP_GRADIENTS
	// allocate memory for output
	int gradWidth = max_x - min_x +2*padX  -2;
	int gradHeight = max_y - min_y +2*padY -2;

	float *h_pGradMag = (float*) malloc(sizeof(float)*2*gradWidth*gradHeight);
	if(!h_pGradMag) {
		printf("h_pGradMag: malloc failed \n");
		return -1;
	}
	// copy results
	cudaMemcpy(h_pGradMag, d_pGradMag, sizeof(float)*2*gradWidth*gradHeight, cudaMemcpyDeviceToHost);

	// write complete output to file
	const int W = 2 * gradWidth;
	FILE* fmag = fopen("gradient_magnitudes.txt", "w");
	FILE* fdir = fopen("gradient_directions.txt", "w");

	if(!fmag) printf("failed to open output file: fmag\n");
	if(!fdir) printf("failed to open output file: fdir\n");

	for(size_t j=0; j < gradHeight; j++) {
	//	fprintf(fmag, "%d\n", j);
	//	fprintf(fdir, "%d\n", j);
		for(size_t i=0; i < 2*gradWidth; i+=2) {
			fprintf(fdir, "%.3f ", h_pGradMag[j*W+i] );			// x - dir
			fprintf(fmag, "%.8f ", h_pGradMag[j*W+i+1] );	 // magnitudes - y
		}
		fprintf(fmag, "\n");
		fprintf(fdir, "\n");
	}
	fclose(fmag);
	fclose(fdir);
#endif

	return 0;
}

