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

#include "nms.h"
#include "global.h"
#include "parameters.h"

#include <math.h>
#include <stdio.h>

#define NMS_MAX_ITERATIONS	100
#define NMS_MODE_EPSILON	((float)1E-5f)
#define NMS_FINAL_DIST		1.f

namespace cudaHOG {

typedef struct {
	float x;
	float y;
	float scale;
	float score;
} NMSDetection;

const float sigma_factor = 4.f;
float sigma_x;
float sigma_y;
float sigma_s;

inline float distance(NMSDetection* p1, NMSDetection* p2)
{
	const float exp_scale = expf(p2->scale);
	float ns[3];
	ns[0] = sigma_x * exp_scale;
	ns[1] = sigma_y * exp_scale;
	ns[2] = sigma_s;

	float b[3];
	b[0] = (p2->x - p1->x) / ns[0];
	b[1] = (p2->y - p1->y) / ns[1];
	b[2] = (p2->scale - p1->scale) / ns[2];

	return b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
}

void nms_process_detections(DetectionList& detections, DetectionList& detections_nms,
							ModelParameters& params)
{
	const int window_width = params.HOG_WINDOW_WIDTH;
	const int window_height = params.HOG_WINDOW_HEIGHT;

	sigma_x = SIGMA_FACTOR;
	sigma_y = (params.HOG_WINDOW_HEIGHT / (float)params.HOG_WINDOW_WIDTH) * sigma_factor;
	sigma_s = logf(SIGMA_SCALE);

// put the results into 3D space (x,y,log(scale))
	NMSDetection nms[4096];
	float nms_score[4096];
	NMSDetection modes[4096];

	NMSDetection ms[4096];

	if( detections.size() > 4096 ) {
		printf("too many detections!\n");
		return;
	}
	int cnt = 0;
	for(size_t i=0; i < detections.size(); i++) {
		Detection* ptr = &(detections[i]);
		nms[cnt].x = ptr->x + floorf((window_width*ptr->scale)) / 2.f;
		nms[cnt].y = ptr->y + floorf((window_height*ptr->scale)) / 2.f;
		nms[cnt].scale = logf(ptr->scale);
		// map scores to positive scores (sigmoid)
		nms_score[cnt] = fmax( /*ptr->scale * */ ptr->score, 0.f );

		cnt++;
	};

	float nx, ny, nz;
	float pi[3];
	float pj[3];
	float numer[3];
	float denum[3];

	for(int i=0; i < cnt; i++) {
		numer[0] = 0.f; numer[1] = 0.f; numer[2] = 0.f;
		denum[0] = 0.f; denum[1] = 0.f; denum[2] = 0.f;

		for(int j=0; j < cnt; j++) {
			float w;
			const float exp_scale = expf(nms[j].scale);
			nx = sigma_x * exp_scale;
			ny = sigma_y * exp_scale;
			nz = sigma_s;

			pi[0] = nms[i].x / nx;
			pi[1] = nms[i].y / ny;
			pi[2] = nms[i].scale / sigma_s;

			pj[0] = nms[j].x / nx;
			pj[1] = nms[j].y / ny;
			pj[2] = nms[j].scale / sigma_s;

			float sqrdist = (pi[0] - pj[0]) * (pi[0] - pj[0]) +
							(pi[1] - pj[1]) * (pi[1] - pj[1]) +
							(pi[2] - pj[2]) * (pi[2] - pj[2]);

			w = nms_score[j] * expf(-sqrdist/2.f) / sqrtf( nx * ny * nz);

			numer[0] += w * pj[0];
			numer[1] += w * pj[1];
			numer[2] += w * pj[2];
			denum[0] += w / nx;
			denum[1] += w / ny;
			denum[2] += w / nz;
		}

		ms[i].x = numer[0] / denum[0];
		ms[i].y = numer[1] / denum[1];
		ms[i].scale = numer[2] / denum[2];
	}

	// mean shift -- iteratively move the points to mode
	for(int i=0; i < cnt; i++) {
		NMSDetection point;
		NMSDetection moved_point;

		moved_point.x = ms[i].x;
		moved_point.y = ms[i].y;
		moved_point.scale = ms[i].scale;

		int count = 0;
		do {

			point.x = moved_point.x;
			point.y = moved_point.y;
			point.scale = moved_point.scale;

			float n[3] = {0,0,0};
			float d[3] = {0,0,0};

			for(int j=0; j < cnt; j++) {
				float w;
				const float exp_scale = expf(nms[j].scale);
				nx = sigma_x * exp_scale;
				ny = sigma_y * exp_scale;
				nz = sigma_s;

				pi[0] = point.x / nx;
				pi[1] = point.y / ny;
				pi[2] = point.scale / sigma_s;

				pj[0] = nms[j].x / nx;
				pj[1] = nms[j].y / ny;
				pj[2] = nms[j].scale / sigma_s;

				float sqrdist = (pi[0] - pj[0]) * (pi[0] - pj[0]) +
								(pi[1] - pj[1]) * (pi[1] - pj[1]) +
								(pi[2] - pj[2]) * (pi[2] - pj[2]);
				w = nms_score[j] * expf(-sqrdist/2.f) / sqrtf( nx * ny * nz);

				n[0] += w * pj[0];
				n[1] += w * pj[1];
				n[2] += w * pj[2];
				d[0] += w / nx;
				d[1] += w / ny;
				d[2] += w / nz;
			}

			moved_point.x = n[0] / d[0];
			moved_point.y = n[1] / d[1];
			moved_point.scale = n[2] / d[2];

			count++;
		} while(  ( count < NMS_MAX_ITERATIONS )
				&&( distance(&point, &moved_point) > NMS_MODE_EPSILON ) );

		// save the mode this point moved to
		modes[i].x = moved_point.x;
		modes[i].y = moved_point.y;
		modes[i].scale = moved_point.scale;
		modes[i].score = nms_score[i];
	}


	NMSDetection nmsModes[4096];
	int nValidModes =0;
	// extract the valid modes from modes array (output
#if NMS_MAXIMUM_SCORE == 0
	// weighted sum score value for each mode!
	for(int i=0; i < cnt; i++) {
		int include = 1;
		for(int j=0; j < nValidModes; j++) {
			if( distance( &(nmsModes[j]), &(modes[i]) ) < NMS_FINAL_DIST) {
				include = 0;
				break;
			}
		}

		if( include ) {
			nmsModes[nValidModes].x = modes[i].x;
			nmsModes[nValidModes].y = modes[i].y;
			nmsModes[nValidModes].scale = modes[i].scale;

			nValidModes++;
		}

	}
	// find score for each valid mode
	for(int i=0; i < nValidModes; i++) {
		float average = 0.f;
		for(int j=0; j < cnt; j++) {
			const float exp_scale = expf(nms[j].scale);
			nx = sigma_x * exp_scale;
			ny = sigma_y * exp_scale;
			nz = sigma_s;

			float p[3];
			p[0] = (nms[j].x - nmsModes[i].x) / nx;
			p[1] = (nms[j].y - nmsModes[i].y) / ny;
			p[2] = (nms[j].scale - nmsModes[i].scale) / nz;
			float sqrdist = p[0]*p[0] +  p[1]*p[1] + p[2]*p[2];

			average += nms_score[j] * expf(-sqrdist/2.f)/sqrtf(nx * ny * nz);
		}
		// convert result and put into final DetectionList
		float scale = expf(nmsModes[i].scale);
		int x = (int) ceilf( nmsModes[i].x - window_width*scale / 2.f );
		int y = (int) ceilf( nmsModes[i].y - window_height*scale / 2.f );

		detections_nms.push_back(Detection(x,y,scale,average);
	}
#else
	// maximum score in each mode becomes overall score of mode
	for(int i=0; i < cnt; i++) {
		int include = 1;
		for(int j=0; j < nValidModes; j++) {

			NMSDetection *p1 = &(nmsModes[j]);
			NMSDetection *p2 = &(modes[i]);

			// compute distance -- with special mean of the two scales!
				const float exp_scale = (expf(p2->scale) + expf(p1->scale) ) / 2.f;
				float ns[3];
				ns[0] = sigma_x * exp_scale;
				ns[1] = sigma_y * exp_scale;
				ns[2] = sigma_s;

				float b[3];
				b[0] = (p2->x - p1->x) / ns[0];
				b[1] = (p2->y - p1->y) / ns[1];
				b[2] = (p2->scale - p1->scale) / ns[2];

				float dist = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];

			if( dist <= NMS_FINAL_DIST) {
				include = 0;
				if( nmsModes[j].score < modes[i].score ) {
					nmsModes[j].score = modes[i].score;
				}
				break;
			}
		}
		if( include ) {
			nmsModes[nValidModes].x = modes[i].x;
			nmsModes[nValidModes].y = modes[i].y;
			nmsModes[nValidModes].scale = modes[i].scale;
			nmsModes[nValidModes].score = modes[i].score;

			nValidModes++;
		}

	}
	// find score for each valid mode
	for(int i=0; i < nValidModes; i++) {
		// convert result and put into final DetectionList
		float scale = expf(nmsModes[i].scale);
		int x = (int) ceilf( nmsModes[i].x - window_width*scale / 2.f );
		int y = (int) ceilf( nmsModes[i].y - window_height*scale / 2.f );
		detections_nms.push_back(Detection(x, y, scale, nmsModes[i].score));
	}
#endif

}

void nms_test(const char* fnDetections)
{
/*
	FILE* fp = fopen(fnDetections, "r");

	int line_count = 0;
	int expected;
	float score, scale;
	int w,h,x,y,origx,origy;
	DetectionList detections;
	detections_init(&detections);

	fscanf(fp, "%d\n", &expected);
	while(8 == fscanf(fp, "%f %f %d %d %d %d %d %d\n", &scale, &score, &w, &h, &x, &y, &origx, &origy) )
	{
		detections_add(&detections, x, y, scale, score);
		line_count++;
	}
	if(line_count != expected ) {
		printf("WARNING: expected %d detections, but only read %d\n", expected, line_count);
		return;
	}

	DetectionList nms_detections;
	detections_init(&nms_detections);

	nms_process_detections(&detections, &nms_detections);

	detections_dump(&nms_detections, "test_detections_nms.txt");
*/

}

}	// end of namespace cudaHOG
