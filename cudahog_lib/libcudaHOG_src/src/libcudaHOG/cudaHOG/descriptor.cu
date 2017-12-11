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

#include "global.h"
#include "parameters.h"

namespace cudaHOG {

// create block descriptor with x,y being the upper-left block
int compute_descriptor(float* d_pDescriptor, float* d_pBlocks, int x, int y, int nBlocksPerRow, ModelParameters& params)
{
	const int nPerBlock = HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS;
	const int nBytesPerBlock = sizeof(float) * nPerBlock;

	for(int i=0; i < params.HOG_DESCRIPTOR_HEIGHT; i++) {
			cudaMemcpy(d_pDescriptor + nPerBlock * (i * params.HOG_DESCRIPTOR_WIDTH),
						d_pBlocks + nPerBlock * ((i+y) * nBlocksPerRow + x),
						nBytesPerBlock * params.HOG_DESCRIPTOR_WIDTH,
						cudaMemcpyDeviceToDevice);
	}
	return 0;
}

}	// end of namespace cudaHOG
