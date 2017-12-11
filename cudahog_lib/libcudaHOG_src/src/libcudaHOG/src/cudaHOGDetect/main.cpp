//
//    Copyright (c) 2011
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
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <QImage>
//#include <types.h>
#include <boost/program_options.hpp>
#ifndef WIN32
#include <dirent.h>
#endif
namespace po = boost::program_options;

#include "cudaHOG.h"

using namespace cudaHOG;

static const char* __usage__ =
" usage: cudaHOGDetect "\
" --image <image_file>"\
" --list <image_list_txt>"\
" [--output <output_directory>]"\
" [--model <SVM model file>] (default: 'dalal')"\
"\n\n"\
" --directory (-d)\t\ta directory of image files to process\n"\
" --image (-i)\t\timage file to run the detector on\n"\
" --list (-l)\t\ta list text file with the filenames of images one line each\n"\
" optional:\n"\
" --output (-o)\t\tthe directory where all the detection file go\n"\
" --model (-m)\t\tthe model file to use (it has to be a binary format 'svmdense' model)\n"\
" --groundplane (-g)\t\tspecify a file with camera and groundplane data\n"\
" --hwmin (-a)\t\tminimum object height (1400)\n"\
" --hwmax (-b)\t\tmaximum object height (2200)\n"\
" --width (-w)\t\tconstant width (in pixels) in the center of the image to use for ROI\n"\
" --homography (-y)\t\tfile containing the groundplane to image homography\n"\
" --minheight\t\tminimum object height in pixels in the image - smaller objects will not be found\n"\
" --maxheight\t\tmax object height in pixels in the image - smaller objects will not be found\n"\
" --config\t\tmultiple model configuration file (for multi-class detection)\n"\
" --active\t\tuse only this one model from the config file for detection\n"\
" --startscale (-s)\t\tthe first scale to scan, default 1.0\n"\
" --scalefactor (-f)\t\tmultiplicative scale step, default 1.05\n"
;

#ifndef WIN32
int list_image_files(std::vector<std::string>& fnImages, const std::string& directory)
{
	DIR *dp;
		struct dirent *p_dirent;
		dp = opendir(directory.c_str());
		if(!dp) {
			printf("could not find directory: %s\n", directory.c_str());
			return -1;
		}

		while( (p_dirent = readdir(dp)) ) {
			if( p_dirent->d_type == DT_REG || p_dirent->d_type == DT_LNK) {
				string image(p_dirent->d_name);
				// only take files with .png or .jpeg or .jpg extension
				string ext = image.substr(image.rfind("."));
				if(!ext.compare(".jpg") || !ext.compare(".png")) {
					fnImages.push_back(image);
				}
				else {
					printf("skipping: %s\n", image.c_str());
				}
			}
		}
		closedir(dp);
	
	return 0;
}
#else
#include <io.h>
#include <windows.h>

int list_image_files(std::vector<std::string>& fnImages, const std::string& directory) 
{
	HANDLE hSearch;
	WIN32_FIND_DATA FileInfo;
	std::string pattern(directory);
	pattern.append("\\*");
	
	hSearch = ::FindFirstFile(pattern.c_str(), &FileInfo);
	if(hSearch != INVALID_HANDLE_VALUE) {
		do {
			if(FileInfo.cFileName[0] != '.') {
				string filename(FileInfo.cFileName);
				
				string ext = filename.substr(filename.rfind("."));
				if(!ext.compare(".jpg") || !ext.compare(".png")) {
					fnImages.push_back(std::string(filename));
				}
			}
		} while(::FindNextFile(hSearch, &FileInfo));
	}
	FindClose(hSearch);
	
	return 0;
}


#endif

int main(int argc, char** argv)
{
	char* mybuf = (char*)malloc(1024);
	setvbuf( stdout , mybuf, _IOLBF , 1024 );

	using namespace std;

	const std::string sep = "/";
	string image, output_dir, model, listfile, directory, groundplane, homography, config, active;
	float h_w_min, h_w_max, start_scale, scale_step;
	int minheight, maxheight;
	int constant_roi_width;

	po::options_description opts("options");
	opts.add_options()
		("help,h",
			"usage information...")
		("output,o", po::value(&output_dir)->default_value("."),
			"the directory where the output will be written")
		("image,i", po::value(&image),
			"the directory of images to run detector on")
		("list,l", po::value(&listfile),
		 	"a text file with one image file on each line")
		("model,m", po::value(&model)->default_value("dalal"),
		 	"SVM model to use - binary model file")
		("directory,d", po::value(&directory),
		 	"directory containing image files to be processed")
		("groundplane,g", po::value(&groundplane),
		 	"path to the camera data file - to calculate groundplane information")
		("hwmin,a", po::value(&h_w_min),
		 	"minimum object height in the world")
		("hwmax,b", po::value(&h_w_max),
		 	"maximum object height in the world")
		("width,w", po::value(&constant_roi_width),
			"width of the roi in pixels")
		("homography,y", po::value(&homography),
			"file containing the groundplane to image homography")
		("minheight", po::value(&minheight),
			"minimum height in pixels of an object")
		("maxheight", po::value(&maxheight),
			"maximum height in pixels of an object")
		("config", po::value(&config),
			"multi model configuration file (alternative to --model)")
		("active", po::value(&active),
		 	"only use this model for detection")
		("startscale,s", po::value(&start_scale),
		 	"first scale to scan, default 1.0")
		("scalefactor,f", po::value(&scale_step),
		 	"factor - multiplicative scale step, default 1.05")
	;
	po::positional_options_description pos_opts;
	pos_opts.add("image", 1);
	pos_opts.add("model",  1);
	pos_opts.add("output", 1);

	po::variables_map args;
	try {
		po::store(po::command_line_parser(argc, argv)
							.options(opts)
							.positional(pos_opts).run()
					, args);
	} catch(std::exception& e ) {
		std::cout << e.what() << "\n";
		return 2;
	}

	po::notify(args);
	if( args.count("help")) {
		printf("%s\n", __usage__);
		return 1;
	}
	if( !args.count("model") && !args.count("config")) {
		std::cout << "Please specify an SVM model file to use\n";
		std::cout << "Alternatively, specify a multi model configuration file\n";
		return 1;
	}
	if( !args.count("image")
		&& !args.count("list")
		&& !args.count("directory"))
	{
		std::cout << "Please specify images\n";
		std::cout << "You can do this through either --list or --directory\n";
		return 1;
	}
	if( !args.count("output") ) {
		std::cout << "Please specify a directory for the output\n";
		return 1;
	}
	if( args.count("width") && !args.count("groundplane") ) {
		printf("--width can only be used together with --groundplane\n");
		return -1;
	}
	if( args.count("groundplane") && args.count("homography") ) {
		printf("please specify _either_ --groundplane or --homography\n");
		return -1;
	}
	if( args.count("minheight") && !args.count("image") && !args.count("homography") ) {
		printf("--minheight only makes sense together with --image and --homography currently!\n");
		return -1;
	}
	if( args.count("maxheight") && !args.count("image") && !args.count("homography") ) {
		printf("--maxheight only makes sense together with --image and --homography currently!\n");
		return -1;
	}

	std::string filename = image.substr(image.rfind("/")+1).append(".detections");
	std::string fnDetections = output_dir + sep + filename;

	// -----------------------------------------------------------------------

	if( args.count("image") ) {
	// run on just one image

		printf("loading image: %s\n", image.c_str());

		QImage img;
		img.load(image.c_str());
		if(img.isNull()) {
			printf("loading failed!\n");
			return 1;
		}

		// 1. convert image to ARGB32 (make sure it really is in that format)
		// 2. crop image like the CPU code does.
		// this means that we make the image dimensions a multiple of the cell_size (i.e. 8)
		// probably this is not the best thing to do, since we lose information by doing this!
		// but for comparison with the CPU code results we do it anyway.
//		const int nCellsX = (int)floorf( (img.width()-1) / 8.f );	 // -1 due to gradient calculation
//		const int nCellsY = (int)floorf( (img.height()-1) / 8.f );
//		const int nX = nCellsX * 8;
//		const int nY = nCellsY * 8;
//		QImage croppedImg = img.convertToFormat(QImage::Format_ARGB32).copy(0,0, nX, nY);
		QImage croppedImg = img.convertToFormat(QImage::Format_ARGB32);

		cudaHOGManager hog;

		if( args.count("config") ) {
			if( hog.read_params_file(config) ) {
				std::cout << "An error occured while processing the configuration file\n";
				return -1;
			}
			if( hog.load_svm_models() ) {
				std::cout << "The models could not be initialized\n";
				return -1;
			}
		} else {
			// a single model was specified
			hog.add_svm_model(model);
		}

		if (hog.prepare_image(croppedImg.bits(), croppedImg.width(), croppedImg.height())) {
			return 1;
		}
//#define TEST_DIRECT
#ifndef TEST_DIRECT
		if( args.count("groundplane") ) {
			printf("loading camera and groundplane data\n");
			if( hog.read_camera_data(groundplane) ) {
				printf("failed to read camera data\n");
				return 1;
			}
			if( args.count("width") ) {
				hog.set_roi_x_center_constraint(constant_roi_width);
			}
			if( args.count("hwmin") && args.count("hwmax") ) {
				printf("setting hwmin & hwmax\n");
				hog.set_groundplane_corridor(h_w_min, h_w_max);
			}
			if( hog.prepare_roi_by_groundplane() )
				return 1;
		}
#else
// ------------------------------------------------------------------
// TEST library routines
// ------------------------------------------------------------------
	float cam_K[3][3];
	float cam_R[3][3];
	float cam_t[3];
	float GP_n[3];
	float GP_d;

	ifstream fs;
	fs.open(groundplane.c_str(), ifstream::in);
	if(! fs.good()) {
		return -1;
	}

//	printf("reading camera data...");

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

	for(int i=0; i < 3; i++)
		fs >> GP_n[i];

	fs >> GP_d;
	GP_d *= -1.f;

	hog.set_camera(&cam_R[0][0], &cam_K[0][0], &cam_t[0]);
	hog.set_groundplane(&GP_n[0], &GP_d);
	hog.prepare_roi_by_groundplane();
#endif
// ------------------------------------------------------------------
// ------------------------------------------------------------------
		if( args.count("homography") ) {
			// read homography from file
			// set the homography for the current image
			printf("setting the homography directly...\n");
			if( hog.set_groundplane_homography(homography) ) {
				printf("failed\n");
				return -1;
			}
			else printf("success\n");

			if( args.count("hwmin") && args.count("hwmax") ) {
				printf("setting hwmin & hwmax\n");
				hog.set_groundplane_corridor(h_w_min, h_w_max);
			}

			if( args.count("minheight") && args.count("maxheight") ) {
				printf("setting minheight & maxheight\n");
				hog.set_valid_object_height(minheight, maxheight);
			}
			if( args.count("width") ) {
				hog.set_roi_x_center_constraint(constant_roi_width);
			}

			if( hog.prepare_roi_by_groundplane() ) {
				printf("failed to precompute the ROI\n");
				return -1;
			}

		}

		vector<cudaHOG::Detection> detections;
		if (hog.test_image(detections)) {
			printf("test_image() failed\n");
			return 1;
		}

		printf("\n%d detections\n\n", (int)detections.size());

		vector<cudaHOG::Detection>::iterator it;
		for(it=detections.begin(); it != detections.end(); it++) {
			printf("%d %d\t%f\t%f\n", it->x, it->y, it->scale, it->score);
		}

		return 0;
	}
	else if( args.count("list") ) {
	// run on a whole list of images
	// prepare two vectors, images files and output filenames
		vector<string> fnImages;
		vector<string> fnDetections;

		ifstream s(listfile.c_str(), ifstream::in);

		while( s >> image ) {
			// strip whitespace from beginning of line
			image = image.substr(image.find_first_not_of(' '));

			if( image.at(0) == '#' )	// skip comment lines
				continue;
			fnImages.push_back(image);

			std::string filename = image.substr(image.rfind("/")+1).append(".detections");
			fnDetections.push_back(output_dir + sep + filename);
		}
		s.close();

		printf("#images to process: %d\n", (int)fnImages.size());
		cudaHOGManager hog(model);
		if( hog.test_images(fnImages, fnDetections) ) {
			printf("ERROR: test_images failed\n");
			return -1;
		}
	}
	else if( args.count("directory") ) {
	// run on all files within a directory
		vector<string> fnImages;
		vector<string> fnDetections;

		if( list_image_files(fnImages, directory) )
			return -1;

		for(unsigned int dd=0; dd < fnImages.size(); dd++) {
			string output = fnImages[dd].substr(fnImages[dd].rfind("/")+1).append(".detections");
			fnDetections.push_back(output_dir + sep + output);
			fnImages[dd] = directory + sep + fnImages[dd];
		}				
		
		printf("#images to process: %d\n", (int)fnImages.size());

		cudaHOGManager hog;
		if( args.count("config") ) {
			if( hog.read_params_file(config) ) {
				std::cout << "An error occured while processing the configuration file\n";
				return -1;
			}
			if( args.count("active") ) {
				printf("selecting active model: %s\n", active.c_str());
				if( hog.set_active_model( active ) ) {
					printf("failed to choose active model\n");
					return -1;
				}
			}
			else {
				if( hog.load_svm_models() ) {
					std::cout << "The models could not be initialized\n";
					return -1;
				}
			}
		} else {
			// a single model was specified
			hog.add_svm_model(model);
		}

		if( 	(args.count("scalefactor") && !args.count("startscale"))
			||  (!args.count("scalefactor") && args.count("startscale")) ) {
			printf("please specify BOTH scalefactor (-f) and startscale (-s)\n\n");
			return -1;
		}
		if( args.count("scalefactor") && args.count("startscale") ) {
			printf("scalefactor: %f\tstartscale: %f\n", args.count("scalefactor"), args.count("startscale"));
			hog.set_detector_params( start_scale, scale_step );
		}

		if( args.count("groundplane") ) {
			printf("enabling ROI computation by groundplane...\n");
			hog.set_camera_data_path(groundplane, "camera.%05d");

			if( (args.count("hwmin") && !args.count("hwmax") )
				||(!args.count("hwmin") && args.count("hwmax") ) ) {
				printf("Specify either both or neither --hwmin and --hwmax\n");
				return -1;
			}
			if( args.count("hwmin") && args.count("hwmax") ) {
				hog.set_groundplane_corridor(h_w_min, h_w_max);
			}
			if( args.count("width") ) {
				hog.set_roi_x_center_constraint(constant_roi_width);
			}
		}

		int cntBlocks = 0;
		int cntSVM = 0;

		double timings[5] = {0.0,0.0,0.0,0.0,0.0};

		if( hog.test_images(fnImages, fnDetections, &cntBlocks, &cntSVM, timings) ) {
			printf("ERROR: test_images failed\n");
			return -1;
		}

		printf("\nstatistics:\n");
		printf("Blocks:\t%d\n", cntBlocks);
		printf("SVM:\t%d\n", cntSVM);

		printf("gradients:\t %f\n", timings[0] / fnImages.size());
		printf("blocks:\t\t %f\n", timings[1] / fnImages.size());
		printf("svm:\t\t %f\n", timings[2] / fnImages.size());
		printf("nms:\t\t %f\n", timings[3] / fnImages.size());
		printf("\ntime per frame:\t %f\n\n", timings[4] / fnImages.size());

		FILE* fps = fopen("statistics.txt", "w");
		fprintf(fps, "Blocks:\t%d\n", cntBlocks);
		fprintf(fps, "SVM:\t%d\n", cntSVM);

		fprintf(fps, "per Frame:\n");
		fprintf(fps, "Blocks:\t%d\n", (int)floorf(cntBlocks / fnImages.size() +0.5f) );
		fprintf(fps, "SVM:\t%d\n", (int)floorf(cntSVM / fnImages.size() +0.5f) );

		fprintf(fps, "\ngradients:\t %f\n", timings[0] / fnImages.size());
		fprintf(fps, "blocks:\t\t %f\n", timings[1] / fnImages.size());
		fprintf(fps, "svm:\t\t %f\n", timings[2] / fnImages.size());
		fprintf(fps, "nms:\t\t %f\n", timings[3] / fnImages.size());
		fprintf(fps, "\ntime per frame:\t %f\n\n", timings[4] / fnImages.size());
		fclose(fps);
	}
	return 0;
}
