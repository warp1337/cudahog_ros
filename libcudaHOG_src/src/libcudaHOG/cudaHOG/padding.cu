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


#include <stdio.h>	// only for test routine!

#include "global.h"

// pad the image in host memory while copying it over to the device
int pad_image(uchar4** d_pPaddedImg, uchar4* h_pImg,
				int width, int height, int padX,int padY)
{
	cudaError_t e;

	int paddedWidth = width + 2 * padX;
	int paddedHeight = height + 2 * padY;

	cudaMalloc((void**)d_pPaddedImg, sizeof(uchar4) * paddedWidth * paddedHeight);
		e = cudaGetLastError();
		if(e) { return -1; }

	cudaMemset(*d_pPaddedImg, 0, sizeof(uchar4) * paddedWidth * paddedHeight);
		e = cudaGetLastError();
		if(e) { return -2; }

	cudaMemcpy2D(*d_pPaddedImg + padX + paddedWidth * padY, 	// dst
				paddedWidth * sizeof(uchar4),					// dpitch
				h_pImg, 										// src
				width * sizeof(uchar4),							// spitch
				width * sizeof(uchar4), 						// width (of transfer matrix)
				height,											// height (of transfer matrix)
				cudaMemcpyHostToDevice);						// copy type
		e = cudaGetLastError();
		if(e) { return -3; }

	// repeat image borders
	// top
	for(int i=0; i < padY; i++) {
		cudaMemcpy( (*d_pPaddedImg) + padX + i * paddedWidth,
					(*d_pPaddedImg) + padX + padY * paddedWidth,
					sizeof(uchar4) * width,
					cudaMemcpyDeviceToDevice);
	}
	// bottom
	for(int i=0; i < padY; i++) {
		cudaMemcpy( (*d_pPaddedImg)+ (padY+height+i) * paddedWidth,
					(*d_pPaddedImg)+ (padY+height-1) * paddedWidth,
					sizeof(uchar4) * paddedWidth,
					cudaMemcpyDeviceToDevice);
		e = cudaGetLastError();
		if(e) { return -7; }
	}

	// left
	uchar4* leftBorder = (uchar4*)malloc(sizeof(uchar4) * paddedHeight );
	cudaMemcpy2D(leftBorder,					 	// dst
				sizeof(uchar4),						// dpitch
				(*d_pPaddedImg) + padX, 			// src
				paddedWidth * sizeof(uchar4),		// spitch
				sizeof(uchar4), 					// width (of transfer matrix)
				paddedHeight,						// height (of transfer matrix)
				cudaMemcpyDeviceToHost);			// copy type
		e = cudaGetLastError();
		if(e) { return -4; }

	for(int i=0; i < padX; i++) {
		cudaMemcpy2D((*d_pPaddedImg) + i,		 	// dst
					paddedWidth * sizeof(uchar4),	// dpitch
					leftBorder, 					// src
					sizeof(uchar4),					// spitch
					sizeof(uchar4), 				// width (of transfer matrix)
					paddedHeight,					// height (of transfer matrix)
					cudaMemcpyHostToDevice);		// copy type
			e = cudaGetLastError();
			if(e) { return -5; }
	}
	free(leftBorder);

	// right
	uchar4* rightBorder = (uchar4*)malloc(sizeof(uchar4) * paddedHeight );
	cudaMemcpy2D(rightBorder,								 	// dst
				sizeof(uchar4),					// dpitch
				(*d_pPaddedImg) + width -2 + padX, 										// src
				paddedWidth * sizeof(uchar4),							// spitch
				sizeof(uchar4), 						// width (of transfer matrix)
				paddedHeight,											// height (of transfer matrix)
				cudaMemcpyDeviceToHost);						// copy type
		e = cudaGetLastError();
		if(e) { return -6; }

	for(int i=0; i < padX; i++) {
		cudaMemcpy2D((*d_pPaddedImg) + padX + width + i,								 	// dst
					paddedWidth * sizeof(uchar4),					// dpitch
					leftBorder, 										// src
					sizeof(uchar4),							// spitch
					sizeof(uchar4), 						// width (of transfer matrix)
					paddedHeight,											// height (of transfer matrix)
					cudaMemcpyHostToDevice);						// copy type
			e = cudaGetLastError();
			if(e) { return -7; }
	}
	free(rightBorder);

	return 0;
}

#ifdef DEBUG_PAD_IMAGE
void test_pad_image(uchar4* d_pPaddedImg, int paddedWidth, int paddedHeight)
{
	uchar4* h_pPaddedImg = (uchar4*)malloc(sizeof(uchar4) * paddedWidth * paddedHeight);
	if(!h_pPaddedImg) {
		printf("test_pad_image: malloc failed\n");
		return;
	}

	cudaMemcpy(h_pPaddedImg, d_pPaddedImg, sizeof(uchar4) * paddedWidth * paddedHeight, cudaMemcpyDeviceToHost);

	FILE* fp = fopen("padded_image.txt","w");
	if(!fp) {
		printf("test_pad_image: failed to open file\n");
		return;
	}
	for(int y=0; y < paddedHeight; y++) {
		fprintf(fp, "%d\n", y);
		for(int x=0; x < paddedWidth; x++) {
			fprintf(fp, "(%d,%d,%d,%d) ",
					h_pPaddedImg[y*paddedWidth + x].x,
					h_pPaddedImg[y*paddedWidth + x].y,
					h_pPaddedImg[y*paddedWidth + x].z,
					h_pPaddedImg[y*paddedWidth + x].w);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

    FILE *fpb;
    fpb = fopen("padded_image.ppm", "wb" );
    fprintf( fpb, "P6\n%d %d\n255\n", paddedWidth, paddedHeight );
    for( int iy = 0; iy < paddedHeight; iy++) {
        for( int ix = 0; ix < paddedWidth; ix++ ) {
			fwrite( h_pPaddedImg + (iy * paddedHeight + ix), 1, 3, fpb );
		}
	}
    fclose( fpb );
}
#endif
