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


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <QImage>
#include <cmath>
#include <deque>
#include <limits.h>
#include <time.h>

#include "cudaHOG.h"
#include "hog.h"
#include "svm.h"
#include "global.h"
#include "parameters.h"

#include "detections.h"
#include "roi.h"

#include "ext/Vector.h"
#include "ext/Matrix.h"

float	HOG_START_SCALE = 1.0f;
float	HOG_SCALE_STEP = 1.05f;

namespace cudaHOG {

Parameters g_params;
std::vector<ROI> g_roi;

const int MAX_ROI_COUNT = 128;

class FalsePositive {
public:
	int image_index;
	int x,y;
	float scale;
};

using namespace std;

cudaHOGManager::cudaHOGManager()
	: bImagePrepared(false), bWithModel(true),
	bValidCamera(false), bValidGroundPlane(false), bValidHomography(false),
	imgWidth(-1), imgHeight(-1),
	m_h_w_min(1400.f), m_h_w_max(2100.f),
	m_roi_center_pixels(-1),
	m_minimum_pixel_height(-1), m_maximum_pixel_height(10000000),
	m_cameraDataPath(""), m_cameraFilenameFormat(NULL),
	m_pActiveModel(NULL)
{
	g_params.models.clear();

	if( hog_initialize() ) {
		std::cerr << "failed to initialize hog datastructures\n";
	}
	if( svm_initialize() ) {

	}
}

cudaHOGManager::cudaHOGManager(const std::string& fn)
	: bImagePrepared(false), bWithModel(true),
	bValidCamera(false), bValidGroundPlane(false), bValidHomography(false),
	imgWidth(-1), imgHeight(-1),
	m_h_w_min(1400.f), m_h_w_max(2100.f),
	m_roi_center_pixels(-1),
	m_minimum_pixel_height(-1), m_maximum_pixel_height(10000000),
	m_cameraDataPath(""), m_cameraFilenameFormat(NULL),
	m_pActiveModel(NULL)
{
	g_params.models.clear();

	if( hog_initialize() ) {
		std::cerr << "failed to initialize hog datastructures\n";
	}

	//	initialize SVM model
	if(	svm_initialize() ) {
		std::cerr << "failed to initialize SVM datastructures\n";
	}
	if( add_svm_model(fn) ) {
		std::cerr << "failed to add SVM model\n";
	}
}

cudaHOGManager::~cudaHOGManager()
{
	if( bImagePrepared ) {
		hog_release_image();
	}
	if( hog_finalize() ) {
		std::cerr << "hog_finalize failed!\n";
	}
}

int cudaHOGManager::read_params_file(std::string& fnParams)
{
	g_params.models.clear();
	return g_params.load_from_file(fnParams);
}

int cudaHOGManager::load_svm_models()
{
	assert(g_params.path.size() > 0 );

	printf("# models: %d\n", g_params.models.size());

	for(size_t ii=0; ii < g_params.models.size(); ii++ ) {
		assert(g_params.models[ii].identifier.size() > 0 );

		string fnModel;
		if( g_params.models[ii].filename.length() > 0 )
			fnModel = g_params.path +string("/")+ g_params.models[ii].filename;
		else
			fnModel = g_params.path +string("/")+ g_params.models[ii].identifier;

		printf("loading model: %s\n", fnModel.c_str());
		if( svm_add_model(fnModel.c_str()) )
			return -1;
	}

	return 0;
}

int cudaHOGManager::set_active_model(std::string& modelIdentifier)
{
	ModelParameters tmp_params;
	int found = 0;
	for(size_t ii=0; ii < g_params.models.size(); ii++) {
		printf("g_params.models[ii].identifier: %s\n", g_params.models[ii].identifier.c_str());
		if( ! modelIdentifier.compare(g_params.models[ii].identifier) ) {
			tmp_params = g_params.models[ii];
			found = 1;
			break;
		}
	}

	if( found ) {
		g_params.models.clear();
		g_params.models.push_back(tmp_params);
		m_pActiveModel = &(g_params.models[0]);

		printf("select model: %s\n", m_pActiveModel->identifier.c_str());
		string fnModel = g_params.path + string("/") + m_pActiveModel->filename;
		svm_add_model(fnModel.c_str());;
		return 0;
	} else {

		printf("Cannot select model %s\tnot found\n", modelIdentifier.c_str());
		printf("available models: \n");
		for(size_t ii=0; ii < g_params.models.size(); ii++ ) {
			printf("%s\t", g_params.models[ii].identifier.c_str());
		}
		return -1;
	}
}

int cudaHOGManager::add_svm_model(const std::string& fnModel)
{
	if( svm_add_model(fnModel.c_str()) ) {
		return -1;
	}
	g_params.models.push_back(ModelParameters());
	return 0;
}

int cudaHOGManager::prepare_image(uchar* pImgData, ushort width, ushort height)
{
	if(bImagePrepared)
		hog_release_image();

if( PRINT_VERBOSE_INFO ) {
	printf("\n\npreparing image: %d x %d\n", width, height);
}
	imgWidth = width;
	imgHeight = height;

	if( hog_transfer_image(pImgData, imgWidth, imgHeight) )
		return -1;
	bImagePrepared = true;

	return 0;
}

int cudaHOGManager::read_camera_data(const string& fnCamera)
{
	ifstream fs;
	fs.open(fnCamera.c_str(), ifstream::in);
	if(! fs.good()) {
		bValidCamera = false;
		return -1;
	}

	for(int i=0; i < 3;i++) {
		for(int j=0; j < 3;j++) {
			fs >> cam_K[i][j];
		}
	}

	float tmp; 	// we dont need the kappa values
	for(int i=0; i < 3;i++)
		fs >> tmp;
	for(int i=0; i < 3;i++) {
		for(int j=0; j < 3;j++) {
			fs >> cam_R[i][j];
		}
	}
	for(int i=0; i < 3;i++)
		fs >> cam_t[i];

	for(int i=0; i < 3; i++) {
		fs >> GP_n[i];
	}
	GP_n[0] *= -1.f;
	GP_n[1] *= -1.f;
	GP_n[2] *= -1.f;

	fs >> GP_d;

	bValidCamera = true;
	bValidGroundPlane = true;

	return 0;
}

int cudaHOGManager::set_camera_data_path(const string& cameraDataPath, char* cameraFilenameFormat)
{
	m_cameraDataPath = cameraDataPath;
	m_cameraFilenameFormat = cameraFilenameFormat;
	return 0;
}

int cudaHOGManager::set_groundplane_homography(const string& fnHomography)
{
	ifstream is(fnHomography.c_str());
	if(is.fail()) {
		printf("Failed to open file: %s\n", fnHomography.c_str());
		bValidHomography = false;
		return -1;
	}

if( PRINT_DEBUG_INFO ) {
	printf("reading homography from file:\n");
}

	for(int j=0; j < 3; j++) {
		for(int i=0; i < 3; i++) {
			is >> m_Homography[i][j];
if( PRINT_DEBUG_INFO ) {
			printf("%f\n", m_Homography[i][j]);
}
			if( is.bad() ) {
				printf("Failure while reading homography from file\n");
				printf("file: %s\n", fnHomography.c_str());
				bValidHomography = false;
				return -2;
			}
		}
	}
	for(int i=0; i < 3; i++) {
		is >> m_ProjectedNormal[i];
if( PRINT_DEBUG_INFO ) {
		printf("%f\n", m_ProjectedNormal[i]);
}
		if( is.bad() ) {
			printf("Failure while reading homography from file\n");
			printf("file: %s\n", fnHomography.c_str());
			bValidHomography = false;
			return -2;
		}
	}

	bValidHomography = true;
	return 0;
}

int cudaHOGManager::set_camera(float *R, float *K, float *t)
{
	if(!R || !K || !t) return -1;

	memcpy(cam_R, R, sizeof(float) * 9);
	memcpy(cam_K, K, sizeof(float) * 9);
	memcpy(cam_t, t, sizeof(float) * 3);
	bValidCamera = true;
	return 0;
}

int cudaHOGManager::set_groundplane(float *n, float* d)
{
	if(!n || !d) return -1;

	memcpy(GP_n, n, sizeof(float) * 3);
	GP_d = *d;
	bValidGroundPlane = true;
	return 0;
}

void cudaHOGManager::set_groundplane_corridor(float minimum_height, float maximum_height)
{
	m_h_w_min = minimum_height;
	m_h_w_max = maximum_height;
}

void cudaHOGManager::set_roi_x_center_constraint(int nPixels)
{
	m_roi_center_pixels = nPixels;
}

void cudaHOGManager::set_valid_object_height(int minimum_pixel_height, int maximum_pixel_height)
{
	m_minimum_pixel_height = minimum_pixel_height;
	m_maximum_pixel_height = maximum_pixel_height;
}

void cudaHOGManager::set_detector_params(float start_scale, float scale_step)
{
	printf("HOG_START_SCALE: %f\nHOG_SCALE_STEP: %f\n", start_scale, scale_step);

	HOG_START_SCALE = start_scale;
	HOG_SCALE_STEP = scale_step;
}

void compute_y_solution(float left, float right, float height, int window_height,
		Matrix<float>& C1, Matrix<float>& Hgp, Vector<float>& vnp,
		float h_w, int& y_l, int& y_r)
{
// ----------------------------------------
// compute D in  x'*D*x = 0
// ----------------------------------------
	// equation: h =  (h_w / h_img) * (np(3) * Hb(2,:) + (h_img * np(3) - np(2) ) * Hb(3,:) );
	Vector<float> Hgp_2(3), Hgp_3(3);
	Hgp.getRow(Hgp_2,1);
	Hgp.getRow(Hgp_3,2);

	Hgp_2 *= vnp(2);
	Hgp_3 *= (window_height * vnp(2) - vnp(1));
	Hgp_2 += Hgp_3;
	Vector<float> a(Hgp_2);
	a *= ((float)h_w) / ((float)window_height);
	a(0) = a(0) / 2.f;
	a(1) = a(1) / 2.f;

	Matrix<float> D(C1);
	D(2,0) += a(0);
	D(2,1) += a(1);
	D(2,2) += a(2);
	D(0,2) += a(0);
	D(1,2) += a(1);

// ----------------------------------------
// project x'*D*x = 0 back to image plane -> E
// ----------------------------------------
	// equation: E = inv(Hgp') * D * inv(Hgp)
	Matrix<float> invHgp(Hgp); invHgp.inv();
	Matrix<float> invHgp_tr(invHgp); invHgp_tr.transposed();
	Matrix<float> E(invHgp_tr);
	E *= D;
	E *= invHgp;

// ----------------------------------------
// compute two solution for left screen border
// ----------------------------------------
	float yl_0, yl_1;
	float x = left;
	float p = (E(1,0) * x + E(2,1) ) / E(1,1);

	assert( p*p >= ((E(0,0)*x + E(2,2) + 2*E(2,0)*x) ));
	assert( E(1,1) != 0.f );
    assert( p*p - ((E(0,0)*x + E(2,2) + 2*E(2,0)*x) / E(1,1)) >= 0 );

	float root = sqrtf( p*p - ((E(0,0)*x + E(2,2) + 2*E(2,0)*x) / E(1,1)));
	yl_0 = -p + root;
	yl_1 = -p - root;

	if( abs(yl_0) < abs(yl_1) )
		y_l = yl_0;
	else
		y_l = yl_1;
	y_l = (int)floorf(y_l + 0.5f);

// ----------------------------------------
// compute two solution for right screen border
// ----------------------------------------
	float yr_0, yr_1;
	x = right;
	p = (E(1,0) * x + E(2,1) ) / E(1,1);

    assert( p*p - ((E(0,0)*x + E(2,2) + 2*E(2,0)*x) / E(1,1)) >= 0 );

	root = sqrtf( p*p - ((E(0,0)*x + E(2,2) + 2*E(2,0)*x) / E(1,1)));
	yr_0 = -p + root;
	yr_1 = -p - root;

	if( abs(yr_0) < abs(yr_1) )
		y_r = yr_0;
	else
		y_r = yr_1;
	y_r = (int)floorf(y_r + 0.5f);

	return;
}

// c  = a x b
void cross(Vector<float>& a, Vector<float>& b, Vector<float>& c)
{
	c(0) = a(1)*b(2) - a(2)*b(1);
	c(1) = a(2)*b(0) - a(0)*b(2);
	c(2) = a(0)*b(1) - a(1)*b(0);
}

int cudaHOGManager::compute_roi_one_scale(float scale, ModelParameters& params,
										int& min_x, int& min_y, int& max_x, int& max_y)
{
	Vector<float> n(3);
	n(0) = GP_n[0]; n(1) = GP_n[1]; n(2) = GP_n[2];

	Matrix<float> Hgp(3,3);
	for(int j=0; j < 3; j++) {
		for(int i=0; i < 3; i++) {
			Hgp(i,j) = m_Homography[i][j];
		}
	}



	Vector<float> vnp(3);
	vnp(0) = m_ProjectedNormal[0];
	vnp(1) = m_ProjectedNormal[1];
	vnp(2) = m_ProjectedNormal[2];

	// adapt Homography for the current scale!
	// and
	// adapt last component of normal for current scale!
	Hgp(0,2) = Hgp(0,2) * scale;
	Hgp(1,2) = Hgp(1,2) * scale;
	Hgp(2,2) = Hgp(2,2) * scale;

	vnp(2) *= scale;

	// equation: C1 = Hgb(3,:)' * Hgp(3,:)
	Matrix<float> C1(3,3);
	C1(0,0) = Hgp(0,2) * Hgp(0,2);
	C1(0,1) = Hgp(0,2) * Hgp(1,2);
	C1(0,2) = Hgp(0,2) * Hgp(2,2);
	C1(1,0) = Hgp(1,2) * Hgp(0,2);
	C1(1,1) = Hgp(1,2) * Hgp(1,2);
	C1(1,2) = Hgp(1,2) * Hgp(2,2);
	C1(2,0) = Hgp(2,2) * Hgp(0,2);
	C1(2,1) = Hgp(2,2) * Hgp(1,2);
	C1(2,2) = Hgp(2,2) * Hgp(2,2);

	int min_height_y_l, min_height_y_r, max_height_y_l, max_height_y_r;

	if( m_roi_center_pixels > 0 ) {
		// take a constant number of pixels in the center of the image
		int curwidth = (int)(imgWidth/scale);
		if( curwidth <= m_roi_center_pixels ) {
			min_x = 0;
			max_x = curwidth;
		} else {
			min_x = (curwidth - m_roi_center_pixels ) / 2;
			max_x = min_x + m_roi_center_pixels;
		}
	} else {
		min_x = 0;
		max_x = (int)(imgWidth/scale);
	}

	compute_y_solution((float)min_x, imgWidth / scale, imgHeight / scale, params.HOG_WINDOW_HEIGHT,
						C1, Hgp, vnp, m_h_w_min, min_height_y_l, min_height_y_r);
	compute_y_solution((float)min_x, imgWidth / scale, imgHeight / scale, params.HOG_WINDOW_HEIGHT,
						C1, Hgp, vnp, m_h_w_max, max_height_y_l, max_height_y_r);

	min_y = min( max_height_y_l, max_height_y_r);
	max_y = max( min_height_y_l, min_height_y_r);

	assert( min_y <= max_y );
	if( min_y > max_y ) {
		// invalid ROI
		min_x =INT_MAX; min_y =INT_MAX; max_x =INT_MIN; max_y =INT_MIN;
		return 1;
	}

	// the minimal y value is the minimal footpoint position - so we need to subtract allow for a full bounding
	// box above this footpoint - thus, subtract HOG_WINDOW_HEIGHT
 	min_y -= params.HOG_WINDOW_HEIGHT;

	// round to multiple of cell size
	min_y = HOG_CELL_SIZE * ( floorf(min_y / ((float)HOG_CELL_SIZE)) ) - 1;
	max_y = HOG_CELL_SIZE * ( floorf((max_y + HOG_CELL_SIZE/2.f) / ((float)HOG_CELL_SIZE))) + 1;

	min_y = max(0,min_y);
	max_y = min(imgHeight/scale+2*HOG_PADDING_Y,(float)max_y);
	min_x = max(0,min_x);
	max_x = min(imgWidth/scale+2*HOG_PADDING_X,(float)max_x);

	if(		(max_y - min_y) -2+ 2*HOG_PADDING_Y < params.HOG_WINDOW_HEIGHT
		||	(params.HOG_WINDOW_HEIGHT * scale) < m_minimum_pixel_height
		||	(params.HOG_WINDOW_HEIGHT * scale) > m_maximum_pixel_height
	) {
		min_x =INT_MAX; min_y =INT_MAX; max_x =INT_MIN; max_y =INT_MIN;
		return 1;
	}
	return 0;
}

void merge_roi_vector(std::vector<ROI> tmp_roi, ROI* pROI)
{
	int min_x = INT_MAX, min_y = INT_MAX;
	int max_x = INT_MIN, max_y = INT_MIN;
	for(size_t ii=0; ii < tmp_roi.size(); ii++) {
		min_x = min(tmp_roi[ii].min_x, min_x);
		min_y = min(tmp_roi[ii].min_y, min_y);
		max_x = max(tmp_roi[ii].max_x, max_x);
		max_y = max(tmp_roi[ii].max_y, max_y);
	}
	pROI->min_x = min_x; pROI->min_y = min_y;
	pROI->max_x = max_x; pROI->max_y = max_y;
}

int cudaHOGManager::set_roi_external(std::vector<ROI>& roi)
{
	g_roi.clear();
	g_roi = roi;
	return 0;
}

int cudaHOGManager::prepare_roi_by_groundplane()
{
	if( (!bValidCamera || !bValidGroundPlane) && (!bValidHomography) ) {
		printf("Cannot calculate ROI without either the camera calibration and groundplane"\
				"data, or a direct groundplane homography!\n");
		return -1;
	}

	if(bValidCamera && bValidGroundPlane) {
		if( compute_homography() ) {
			printf("failed to precompute the homography\n");
			return -1;
		}
	}

	g_roi.clear();

	// for each scale compute the ROI
	int width = imgWidth;
	int height = imgHeight;
	float startScale = HOG_START_SCALE;
	int max_window_width = g_params.max_window_width();
	int max_window_height = g_params.max_window_height();
	float endScale = min( ( width + 2*HOG_PADDING_X ) / (float)max_window_width ,
							(height+ 2*HOG_PADDING_Y ) / (float)max_window_height );
	float scale = startScale;
	int count = 0;

	while( 	scale < endScale && count < MAX_ROI_COUNT ) {
		// compute the ROI for each model, at this scale
		// as a result create a ROI that spans all individual ROIs

		std::vector<ROI> tmp_roi;
		for(int ii=0; ii < g_params.models.size(); ii++) {
			ROI roi;
			int min_x, min_y, max_y, max_x;

			// can we skip this model due to scale constraints in config file?
			if(    scale > g_params.models[ii].max_scale
				|| scale < g_params.models[ii].min_scale ) {
				continue;
			}

			if( 1 == compute_roi_one_scale(scale, g_params.models[ii], min_x, min_y, max_x, max_y) ) {
				continue;
			}
			roi.min_x = min_x;
			roi.min_y = min_y;
			roi.max_x = max_x;
			roi.max_y = max_y;
			tmp_roi.push_back(roi);
		}

		// this is a bit hacky, but we want to avoid to expose ROI in the interface, for now
		ROI newROI;

		if( tmp_roi.size() == 0 ) {
			if(DEBUG_PRINT_SCALE_CONSTRAINTS)
				printf("no object category made sense at this scale! skipping\n");
			newROI.min_x = INT_MAX;
			newROI.min_y = INT_MAX;
			newROI.max_x = INT_MIN;
			newROI.max_y = INT_MIN;
		} else if( tmp_roi.size() == 1 ) {
			// no merge required only one ROI, just copy
			newROI.min_x = tmp_roi[0].min_x;
			newROI.min_y = tmp_roi[0].min_y;
			newROI.max_x = tmp_roi[0].max_x;
			newROI.max_y = tmp_roi[0].max_y;
		} else {
			merge_roi_vector(tmp_roi, &newROI);
		}
		newROI.scale = scale;
		g_roi.push_back(newROI);

		scale *= (float)(HOG_SCALE_STEP);
		++count;
	}

if( PRINT_VERBOSE_INFO ) {
	printf("precomputed ROI for %d scales\n", count);
}

	return 0;
}

int cudaHOGManager::compute_homography()
{
	// check that the camera data is available (call prepare_image_camera before this)
	if(!bValidCamera || !bValidGroundPlane) {
		printf("cannot prepare homography without camera & groundplane data\n");
		return -1;
	}
	if(!bImagePrepared) {
		printf("cannot calculate ROI without an actual image\n");
		printf("we need to know about the image size, etc.\n");
		return -2;
	}

	Vector<float> n(3);
	n(0) = GP_n[0]; n(1) = GP_n[1]; n(2) = GP_n[2];

	Matrix<float> R(3,3,(float*)cam_R);
	R.transposed();
	Matrix<float> K(3,3,(float*)cam_K);
	Vector<float> t(3);
	t(0) = cam_t[0]; t(1) = cam_t[1]; t(2) = cam_t[2];

	Matrix<float> KR(K);
	KR *= R;

	Matrix<float> mKRt(KR);
	mKRt *= t;
	Vector<float> KRt(3);
	mKRt.getColumn(KRt,0);

// ----------------------------------------
// compute projection matrix P
// ----------------------------------------
	Matrix<float> P(4,3,0.f);
	for(int i=0; i < 3; i++) {
		for(int j=0; j < 3; j++) {
			P(i,j) = KR(i,j);
		}
	}
	for(int i=0; i < 3; i++) {
		P(3,i) = -KRt(i);
	}
// ----------------------------------------
// compute homography to groundplane - Hgp
// ----------------------------------------
	// compute 3 directional vectors and put them in a matrix
	Vector<float> VPN(3);
	VPN(0) = P(0,2);
	VPN(1) = P(1,2);
	VPN(2) = P(2,2);

	Vector<float> q0(3);
	Vector<float> q1(3);
	Vector<float> q2(3);
	//  operator / is cross product
	Vector<float> tmp(3,0);
	cross(VPN,n,tmp);
	cross(n,tmp,q2);
	cross(n,q2,q1);
	// equation:	q0 = t - ( (n * t) + GP_d ) * n;
	q0 = n;
	q0 *= ( (n * t) + GP_d );
	q0(0) = t(0) - q0(0);
	q0(1) = t(1) - q0(1);
	q0(2) = t(2) - q0(2);

	Matrix<float> Hgp(P);
	Matrix<float> qs(3,4);
	for(int i=0; i < 3; i++) {
		qs(0,i) = q1(i);
		qs(1,i) = q2(i);
		qs(2,i) = q0(i);
	}
	qs(0,3) = 0; qs(1,3) = 0; qs(2,3) = 1;
	// equation: Hgp = P * [q1, q2, q0, (0,0,1)']
	Hgp *= qs;

	for(int j=0; j < 3; j++) {
		for(int i=0; i < 3; i++) {
			m_Homography[i][j] = Hgp(i,j);
		}
	}

	// compute the projected normal
	Matrix<float> np(P);
	Vector<float> nn(4); nn(0) = n(0); nn(1) = n(1); nn(2) = n(2); nn(3) = 0;
	np *= nn;
	Vector<float> vnp(3);
	np.getColumn(vnp,0);

	for(int i=0; i < 3; i++) {
		m_ProjectedNormal[i] = vnp(i);
	}

	return 0;
}

int cudaHOGManager::features_to_file(vector<Feature>& features, const string& fnOutput)
{
	using namespace std;
	// header data
	int version = 0; // does not matter
	int typeid_data = 4; // typeid - 3 == int --- 4 == float
	int typeid_target = 4;
	int count_pos = 0, count_neg = 0;
	for(size_t i=0; i < features.size(); i++) {
		if( features[i].target > 0 ) count_pos++;
		if( features[i].target < 0 ) count_neg++;
	}
	int count_total = count_pos + count_neg;

	ofstream sOutput(fnOutput.c_str(), ios_base::out | ios_base::binary);

	sOutput.write((char*)&version, sizeof(int));
	sOutput.write((char*)&typeid_data, sizeof(int));
	sOutput.write((char*)&typeid_target, sizeof(int));
	sOutput.write((char*)&count_total, sizeof(int));
	int feature_length = features[0].values.size();
	sOutput.write((char*)&(feature_length), sizeof(int));

	for(size_t i=0; i < features.size(); i++) {
		// assume each feature has equal dimension
		sOutput.write((char*)&(features[i].target), sizeof(float));
		for(size_t j=0; j < features[i].values.size(); j++) {
#ifdef WIN32
			assert(! _isnan(features[i].values[j]) );
#else
			assert(! isnan(features[i].values[j]) );
#endif
			sOutput.write((char*)&(features[i].values[j]), sizeof(float));
		}
	}

	printf("wrote %d positives features\n", count_pos);
	printf("wrote %d negative features\n", count_neg);

	sOutput.close();
	return 0;
}

int cudaHOGManager::features_from_file(vector<Feature>& features, const string& fnFeatures)
{
	ifstream s(fnFeatures.c_str(), ios_base::in | ios_base::binary);

	s.seekg(3*sizeof(int), ios_base::beg);
	int count_total;
	s.read((char*)&count_total,sizeof(int));
	s.seekg(5*sizeof(int), ios_base::beg);

	const int dim = m_pActiveModel->dimension();
	for(int i=0; i < count_total; i++) {
		Feature f;
		s.read((char*)&f.target, sizeof(float));
		f.values.resize(dim);
		s.read((char*)&f.values[0], sizeof(float) * dim );
		features.push_back(f);
	}

	int count_pos = 0, count_neg = 0;
	for(size_t i=0; i < features.size(); i++) {
		if( features[i].target > 0 ) count_pos++;
		if( features[i].target < 0 ) count_neg++;
	}
	printf("read %d positive features\n", count_pos);
	printf("read %d negative features\n", count_neg);
	s.close();

	return 0;
}


int cudaHOGManager::release_image()
{
	if(bImagePrepared) {
		if( hog_release_image() )
			return 1;
		bImagePrepared = false;
	}
	return 0;
}


int cudaHOGManager::test_image(DetectionList& detections)
{
	MultiDetectionList all_dets;

	int res = test_image(all_dets);

    if(all_dets.size() > 0)
        detections = all_dets[0];



	return res;
}

int cudaHOGManager::test_image(MultiDetectionList& detections)
{
	if(!bImagePrepared) {
		printf("No image prepared on device\n");
		return 1;
	}
	if(!bWithModel) {
		printf("cudaHOGManager constructed without SVM model file\n");
		return 1;
	}

	int cntBlocks = 0;
	int cntSVM = 0;

	double timings[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

	int res =  hog_process_image_multiscale(imgWidth, imgHeight, g_roi, &cntBlocks, &cntSVM, timings, detections);
	release_image();
    if( PRINT_DEBUG_INFO ) {
        printf("HOG Blocks: %d\n", cntBlocks);
        printf("SVM evaluations: %d\n", cntSVM);
    }

	if( res )
		return 1;
	return 0;
}

int cudaHOGManager::test_images(const std::vector<string>& fnsImages,
								const std::vector<string>& fnsOutput,
								int* cntBlocks,
								int* cntSVM,
								double* timings)
{
	if(!bWithModel) {
		printf("For detection construct cudaHOGManager with SVM model file\n");
		return 1;
	}

	int tcntBlocks = 0;
	int tcntSVM = 0;
	if(cntBlocks == NULL || cntSVM == NULL) {
		cntBlocks = &tcntBlocks;
		cntSVM = &tcntSVM;
	}

	for(size_t i=0; i < fnsImages.size(); i++) {
		QImage img(fnsImages[i].c_str());
		if(img.isNull()) {
			printf("failed to load file: %s\n", fnsImages[i].c_str());
			return 1;
		}
if( DEBUG_PRINT_PROGRESS ) {
		printf("processing image: %s\n", fnsImages[i].c_str());
}
		// convert image to ARGB32 (make sure it really is in that format)
		QImage croppedImg = img.convertToFormat(QImage::Format_ARGB32);

		if( prepare_image(croppedImg.bits(), croppedImg.width(), croppedImg.height()) ) {
			printf("failed to process image: %s\n", fnsImages[i].c_str());
			return 1;
		}

		// optionally, compute ROI now, if a camera path was set
		if( m_cameraDataPath.size() > 0 ) {
			string fnCamera(m_cameraDataPath);
			fnCamera.append("/");

			string digits;
			int idx = fnsImages[i].rfind("_");
			if( idx < 0 ) {
				// no _ found -- try to extract the digits by pattern aaa33333.png
				// take the last 5 digits in front of the extension
				int extension_idx = fnsImages[i].rfind(".");
				string name_without_extension
					= fnsImages[i].substr(0, extension_idx);
				idx = name_without_extension.find_last_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
				digits = name_without_extension.substr( idx+1 );
			}
			else {
				digits = fnsImages[i].substr(idx-5,5);
			}
			fnCamera.append("camera."+digits);

			if( read_camera_data(fnCamera) ) {
				printf("unable to read camera data for file: %s\n", fnsImages[i].c_str());
				printf("was looking here: %s\n", fnCamera.c_str());
				return -1;
			}
			if( prepare_roi_by_groundplane() )
				return -1;
		}

		MultiDetectionList det;
		if( hog_process_image_multiscale(imgWidth, imgHeight, g_roi, cntBlocks, cntSVM, timings, det) ) {
			printf("failed to process image: %s\n", fnsImages[i].c_str());
			return 1;
		}
		dump_MultiDetectionList(det, fnsOutput[i]);
		det.clear();

		hog_release_image();
		bImagePrepared = false;
	}
	return 0;
}


int cudaHOGManager::dump_features(vector<string>& pos, vector<string>& neg,
									bool bSampled, bool bPadding, std::string& fnFeatures)
{
	using namespace std;

	int count_pos = 0, count_neg = 0;
	if( !m_pActiveModel ) {
		printf("no active model selected!\n");
		printf("use set_active_model(..) to choose model parameters.\n");
		return -1;
	}

	vector<Feature> features;

	printf("positive features...\n");
	if( compute_features(pos, count_pos, 1.f, bPadding, features) )
		return -1;
	printf("done.\n");

	if(bSampled) {
		printf("sampling negative features...\n");
		if( compute_features_sampled(neg, count_neg, features) )
			return -2;
	} else {
		printf("negative features...\n");
		if( compute_features(neg, count_neg, -1.f, bPadding, features) )
			return -2;

	}
	printf("done.\n");

	int count_total = count_pos + count_neg;
	printf("generated samples\npositive: %d\nnegative: %d\ntotal:\t%d\n",
			count_pos, count_neg, count_total);
	printf("ratio #neg/#pos: %f\n", (float)count_neg / (float)count_pos);

	return features_to_file(features, fnFeatures);
}


int cudaHOGManager::dump_hard_features(vector<string>& neg, string& fnFeatures, string& fnOutput)
{
	using namespace std;

	if(!bWithModel)
		return 1;

	if( !m_pActiveModel ) {
		printf("no active model selected!\n");
		printf("use set_active_model(..) to choose model parameters.\n");
		return -1;
	}

	vector<Feature> features;
	features_from_file(features, fnFeatures);

	float* descriptor = (float*)malloc(sizeof(float) * m_pActiveModel->dimension() );

	deque<FalsePositive> false_positives;
	for(size_t i=0; i < neg.size(); i++) {
		QImage img(neg[i].c_str());
		if(img.isNull()) {
			printf("failed to load file: %s\n", neg[i].c_str());
			free(descriptor);
			return 1;
		}

	//	printf("processing image: %s\n", neg[i].c_str());

		// 1. convert image to ARGB32 (make sure it really is in that format)
		// 2. crop image like the CPU code does.
		// this means that we make the image dimensions a multiple of the cell_size (i.e. 8)
		// probably this is not the best thing to do, since we lose information by doing this!
		// but for comparison with the CPU code results we do it anyway.
		const int nCellsX = (int)floorf( (img.width()-1) / 8.f );	 // -1 due to gradient calculation
		const int nCellsY = (int)floorf( (img.height()-1) / 8.f );
		const int nX = nCellsX * 8;
		const int nY = nCellsY * 8;
		QImage croppedImg = img.convertToFormat(QImage::Format_ARGB32).copy(0,0, nX, nY);

		if( prepare_image(croppedImg.bits(), croppedImg.width(), croppedImg.height()) ) {
			printf("failed to process image: %s\n", neg[i].c_str());
			free(descriptor);
			return 1;
		}

		int cnt = 0;
		float startScale = 1.0f;
		int window_width = m_pActiveModel->HOG_WINDOW_WIDTH;
		int window_height = m_pActiveModel->HOG_DESCRIPTOR_HEIGHT;
		float endScale = min( ( croppedImg.width() + 2*HOG_PADDING_X ) / (float)window_width ,
								(croppedImg.height()+ 2*HOG_PADDING_Y ) / (float)window_height );
		float scale = startScale;
		while( scale < endScale )
		{
			int curwidth = croppedImg.width() / scale;
			int curheight = croppedImg.height() / scale;

			int padX = HOG_PADDING_X;
			int padY = HOG_PADDING_Y;
//			int padX = 0;
//			int padY = 0;

			MultiDetectionList tmp_false_positives;
			int cntBlocks, cntSVM;	// we actually dont care for the values
			if( hog_process_image(curwidth, curheight, scale, padX, padY,
									NULL, &cntBlocks, &cntSVM,
									tmp_false_positives) )
			{
				printf("failed to process image: %s\n", neg[i].c_str());
				tmp_false_positives.clear();
				hog_release_image();
				free(descriptor);
				return 1;
			}

			// only one model should be loaded!
			// we only want results from this one model!
			assert(tmp_false_positives.size() == 1 );

			for(size_t jj=0; jj < tmp_false_positives[0].size(); jj++) {
				FalsePositive fp;
				Detection d = tmp_false_positives[0][jj];
				fp.x = d.x;
				fp.y = d.y;
				fp.scale = d.scale;
				fp.image_index = i;
				false_positives.push_back(fp);
				++cnt;
			}
			tmp_false_positives.clear();

			scale *= HOG_TRAINING_SCALE_STEP;
		}
//		printf("%#d false-positives in image: %s\n", cnt, neg[i].c_str());
		hog_release_image();
	}

	printf("\nfound %d false positives\n", (int)false_positives.size() );
	printf("\nrandomly selecting up to %d hard examples\n\n", MAXIMUM_HARD_EXAMPLES);

	// randomize the order of false_positives (after that just pick the first element)
	printf("randomizing order of hard examples...\t");
	srand(time(NULL));
	random_shuffle(false_positives.begin(), false_positives.end());
	printf("done.\n");

	int count_neg = 0;
	Feature f;
	f.target = -1.f;
	features.reserve( features.size() + MAXIMUM_HARD_EXAMPLES );
	while( count_neg < MAXIMUM_HARD_EXAMPLES && false_positives.size() != 0 ) {
		deque<FalsePositive>::iterator fp = false_positives.begin();

		QImage img(neg[fp->image_index].c_str());
		if(img.isNull()) {
			printf("failed to load file: %s\n", neg[fp->image_index].c_str());
			free(descriptor);
			return 1;
		}
		const int nCellsX = (int)floorf( (img.width()-1) / 8.f );	 // -1 due to gradient calculation
		const int nCellsY = (int)floorf( (img.height()-1) / 8.f );
		const int nX = nCellsX * 8;
		const int nY = nCellsY * 8;
		QImage croppedImg = img.convertToFormat(QImage::Format_ARGB32).copy(0,0, nX, nY);

		if( prepare_image(croppedImg.bits(), croppedImg.width(), croppedImg.height()) ) {
			printf("failed to process image: %s\n", neg[fp->image_index].c_str());
			free(descriptor);
			return 1;
		}

		int blockX = (fp->x / fp->scale + HOG_PADDING_X ) / HOG_CELL_SIZE;
		int blockY = (fp->y / fp->scale + HOG_PADDING_Y ) / HOG_CELL_SIZE;
	//	int blockX = (fp->x / fp->scale) / HOG_CELL_SIZE;
	//	int blockY = (fp->y / fp->scale) / HOG_CELL_SIZE;
		memset(descriptor,0,sizeof(float) * m_pActiveModel->dimension() );

	//	// enable padding, to also get negative samples with padded image borders
	//	DISABLE padding -- for comparison... apparently enabling padding for hard examples
	//	has negative impact on performance!
		if(	hog_get_descriptor(imgWidth, imgHeight, 1,
							blockX, blockY, fp->scale, *m_pActiveModel, descriptor) ) {
			printf("failed on image: %s\n", neg[fp->image_index].c_str());
			free(descriptor);
			hog_release_image(); bImagePrepared = false;
			return -2;
		}

		f.values.resize(m_pActiveModel->dimension());
		memcpy(&(f.values[0]), descriptor, sizeof(float) * m_pActiveModel->dimension());
		features.push_back(f);
		count_neg++;

		false_positives.pop_front();

		hog_release_image(); bImagePrepared = false;
	}
	printf("done.\n");
	printf("hard examples: %d\n", count_neg);

	false_positives.clear();

	free(descriptor);

	return features_to_file(features, fnOutput);
}

// =========================================================
//			protected
// =========================================================

int cudaHOGManager::compute_features(vector<string>& examples, int& count, float target, bool bPadding,
										vector<Feature>& features)
{
	Feature feat;
	feat.target = target;
	float* descriptor = (float*)malloc(sizeof(float) * m_pActiveModel->dimension() );

	vector<string>::iterator example;
	for(example = examples.begin(); example != examples.end(); example++) {
		QImage img;
		img.load(example->c_str());
		if(img.isNull()) {
			printf("loading failed!\n");
			free(descriptor);
			return -1;
		}

		if( prepare_image(img.bits(), img.width(), img.height()) ) {
			free(descriptor);
			return -1;
		}

		memset(descriptor,0,sizeof(float) * m_pActiveModel->dimension() );
		// by default we do not pad here (offset == 0), INRIA training examples are padded already
		int offset = bPadding ? 0 : 1;

		if(	hog_get_descriptor(imgWidth, imgHeight, offset, 2, 2, 1.f, *m_pActiveModel, descriptor)
			) {
			free(descriptor);
			return -1;
		}

		const int dim = m_pActiveModel->dimension();
		feat.values.resize(dim);
		memcpy(&(feat.values[0]), descriptor, sizeof(float) * dim );
		features.push_back(feat);
		++count;
	}
	free(descriptor);
	return 0;
}


int cudaHOGManager::compute_features_sampled(vector<string>& examples, int& count,
									vector<Feature>& features)
{
	Feature feat;
	feat.target = -1.f;
	typedef Detection Sample;
	float* descriptor = (float*)malloc(sizeof(float) * m_pActiveModel->dimension() );

	srand(time(NULL));

	vector<string>::iterator example;
	for(example = examples.begin(); example != examples.end(); example++) {
		QImage img;
		img.load(example->c_str());
		if(img.isNull()) {
			printf("loading failed: %s\n", example->c_str());
			free(descriptor);
			return -2;
		}

		if( prepare_image(img.bits(), img.width(), img.height()) ) {
			free(descriptor);
			return -3;
		}

		// randomly sample position-scale pairs
		std::vector<Sample> samples;
		const float max_scale_steps_x = logf(img.width() / (float)m_pActiveModel->HOG_WINDOW_WIDTH)
										/ logf(HOG_TRAINING_SCALE_STEP);
		const float max_scale_steps_y = logf(img.height() / (float)m_pActiveModel->HOG_WINDOW_HEIGHT)
										/ logf(HOG_TRAINING_SCALE_STEP);
		const int max_scale_steps = (int)floorf(min(max_scale_steps_x, max_scale_steps_y));

		for(int i=0; i < NEGATIVE_SAMPLES_PER_IMAGE; i++) {
			Sample s;
			s.scale = powf(HOG_TRAINING_SCALE_STEP, rand() % max_scale_steps);

			if(    (img.width() -1 - m_pActiveModel->HOG_WINDOW_WIDTH * s.scale) <= 0
				|| (img.height() -1 - m_pActiveModel->HOG_WINDOW_HEIGHT * s.scale) <= 0 )
			{
				// this should not happen, because we sample only from 'valid' scales
				printf("cannot sample at this scale: %f\n", s.scale);
				i--;
				continue;
			}

			s.x = (rand() % (int)floorf(img.width() -1 - m_pActiveModel->HOG_WINDOW_WIDTH * s.scale));
			s.y = (rand() % (int)floorf(img.height()-1 - m_pActiveModel->HOG_WINDOW_HEIGHT * s.scale));
			//printf("%d\t %d\t %.3f\n", s.x, s.y, s.scale);
			samples.push_back(s);
		}

		// for each sample compute the HOG feature and store it
		for(size_t j=0; j < samples.size(); j++) {
			Sample s = samples[j];
			int blockX = (s.x / s.scale + HOG_PADDING_X) / HOG_CELL_SIZE;
			int blockY = (s.y / s.scale + HOG_PADDING_Y) / HOG_CELL_SIZE;

			memset(descriptor,0,sizeof(float) * m_pActiveModel->dimension() );
			// enable padding, to also get negative samples with padded image borders
			if(	hog_get_descriptor(imgWidth, imgHeight, 1, blockX, blockY,
									s.scale, *m_pActiveModel, descriptor) )
			{
				printf("failed on image: %s\n", example->c_str());
				free(descriptor);
				return -2;
			}

			const int dim = m_pActiveModel->dimension();
			feat.values.resize(dim);
			memcpy(&(feat.values[0]), descriptor, sizeof(float) * dim );

			features.push_back(feat);
			++count;
		}

	}
	free(descriptor);
	return 0;
}

} // namespace cudaHOG
