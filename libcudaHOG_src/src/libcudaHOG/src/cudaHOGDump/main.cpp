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

#include <iostream>
#include <fstream>
#include <QImage>
#include <boost/program_options.hpp>
#ifndef WIN32
#include <dirent.h>
#endif
namespace po = boost::program_options;

//extern "C"
//{
	#include "cudaHOG.h"
//}

using namespace cudaHOG;

using namespace std;

static const char* __usage__ =
"cudaHOGDump --positive <positive dir> --negative <negative dir> [--output <feature file>] \n"\
"\n or \n"\
"cudaHOGDump --negative <negative dir> --features <feature file> --model <svm model file>"\
"\n\n"\
"--positive(-p)			directory with positive example images\n"\
"--negative(-n)			directory with negative example images\n"\
"--sampled(-s)			sample features from negative images - default: 1, set -s 0 for direct negative extraction\n"\
"--output(-o)			file where the output will be written (default: 'features')\n"\
"--features(-f)			file of previously dumped features - hard examples are appended to this file\n"\
"--model(-m)			SVM model to use to find false-positives (in conjunction with false-positives\n\n"\
"--config				pass the HOG model parameters in a file (ini style file, one section for each model)\n"\
"--active(-a)			choose which model to use from the parameters file\n"\
"--extrapadding(-x)		default: true - add extra padding to training example images\n"\
"If --model is specified hard examples will be extracted, instead of the normal negative examples."
;

#ifdef WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#include <string>

int list_files_in_directory(std::vector<std::string>& files, const std::string& directory)
{
	HANDLE hSearch;
	WIN32_FIND_DATA FileInfo;
	std::string pattern(directory);
	pattern.append("\\*");
	
	hSearch = ::FindFirstFile(pattern.c_str(), &FileInfo);
	if(hSearch != INVALID_HANDLE_VALUE) {
		do {
			if(FileInfo.cFileName[0] != '.') {
				string fullpath(directory + "/" + std::string(FileInfo.cFileName));
				files.push_back(fullpath);
			}
		} while(::FindNextFile(hSearch, &FileInfo));
	}
	FindClose(hSearch);

	return 0;
}

#else

int list_files_in_directory(std::vector<std::string>& files, const std::string& directory)
{
	DIR *dp;
	struct dirent *p_dirent;
	dp = opendir(directory.c_str());
	if(!dp) {
		printf("could not find directory: %s\n", directory.c_str());
		return -1;
	}

	const string sep("/");
	while( (p_dirent = readdir(dp)) ) {
		string image(p_dirent->d_name);
		string full_path = directory + sep + image;
		if( p_dirent->d_type == DT_REG || p_dirent->d_type == DT_LNK) {
			files.push_back(full_path);
		} else {
			printf("skipping: neither regular file nor symlink: %s\n", full_path.c_str());
		}
	}
	closedir(dp);

	return 0;
}
#endif

int main(int argc, char** argv)
{
	using namespace std;

	const std::string sep = "/";
	string dirPos, dirNeg, fnOutput, fnFeatures, fnModel, fnParameters, active;
	bool sampledNeg, padding;

	po::options_description opts("options");
	opts.add_options()
		("help,h",
			"usage information...")
		("output,o", po::value(&fnOutput)->default_value("features"),
			"the file where the output will be written (default: 'features')")
		("positive,p", po::value(&dirPos),
		 	"directory with negative example images")
		("negative,n", po::value(&dirNeg),
		 	"directory with positive example images")
		("sampled,s", po::value(&sampledNeg),
			"sample features from negative images")
		("features,f", po::value(&fnFeatures),
			"file of previously dumped features - hard examples will be appended to this file")
		("model,m", po::value(&fnModel),
			"SVM model to use to find false-positives (in conjunction with false-positives")
		("config,c", po::value(&fnParameters),
			"file with parameters for multiple models")
		("active,a", po::value(&active),
			"choose which model to use (has to be specified within the --config file")
		("extrapadding,x", po::value(&padding)->default_value(true),
			"default: true - add extra padding to training example images")
	;
	po::variables_map args;
	try {
		po::store(po::command_line_parser(argc, argv).options(opts).run(), args);
	} catch(std::exception& e ) {
		std::cout << e.what() << "\n";
		return 2;
	}

	po::notify(args);
	if( args.count("help")) {
		printf("%s\n", __usage__);
		return 1;
	}

	if( !args.count("sampled") )
		sampledNeg = true;

	if( args.count("positives") && !args.count("negatives") ) {
		printf("To bootstrap features please specify --positives and --negatives\n");
		printf("Use --help(-h) for usage information\n");
		return 1;
	}

	// generate hard-examples
	if( args.count("negative") && args.count("output") &&
		args.count("features") /*&& args.count("model")*/ && args.count("config") ) {

		if( !args.count("active") ) {
			printf("please use --active(-a) to specify which model to use\n");
			return -1;
		}

		cudaHOGManager hog;

		printf("reading from file: %s\n", fnParameters.c_str());
		if( hog.read_params_file(fnParameters) ) {
			printf("failed to read parameters! Please check file syntax\n");
			return -1;
		}
		if( hog.set_active_model( active ) ) {
			printf("failed to choose active model\n");
			return -1;
		}

		vector<string> neg;
		printf("processing files in directory: %s\n", dirNeg.c_str());
		if( list_files_in_directory(neg, dirNeg) ) return 1;
		printf("# images to check for hard examples: %d\n", (int)neg.size());

		printf("generating hard examples...\n");
		if( hog.dump_hard_features(neg, fnFeatures, fnOutput)) {
			printf("failed to generate hard examples\n");
			return 1;
		}

		return 0;
	}


	// generate first round bootstrap training examples
	if( args.count("positive") && args.count("negative") && !args.count("model") ) {
		// dump features (positive and negatives)
		cudaHOGManager hog;
		vector<string> pos;
		vector<string> neg;

		if( args.count("config") ) {
			if( !args.count("active") ) {
				printf("please use --active(-a) to specify which model to use\n");
				return -1;
			}

			printf("reading from file: %s\n", fnParameters.c_str());
			if( hog.read_params_file(fnParameters) ) {
				printf("failed to read parameters! Please check file syntax\n");
				return -1;
			}
			if( hog.set_active_model( active ) ) {
				printf("failed to choose active model\n");
				return -1;
			}
			printf("chosen active model: %s\n", active.c_str());
			printf("done.\n");
		}

		printf("processing files in directory: %s\n", dirPos.c_str());
		if( list_files_in_directory(pos, dirPos) )
			return 1;
		printf("processing files in directory: %s\n", dirNeg.c_str());
		if( list_files_in_directory(neg, dirNeg) )
			return 1;

		printf("# pos images to process: %d\n", (int)pos.size());
		printf("# neg images to process: %d\n", (int)neg.size());
		printf("sampling from neg images: %s\n", sampledNeg ? "true" : "false");

		if( hog.dump_features(pos, neg, sampledNeg, padding, fnOutput) ) {
			printf("failed\n");
			return 1;
		}
		return 0;
	}

	return 0;
}
