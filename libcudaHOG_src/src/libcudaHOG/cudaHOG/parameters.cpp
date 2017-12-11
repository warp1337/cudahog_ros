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


#include <cassert>
#include <climits>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

#include "global.h"
#include "parameters.h"

namespace cudaHOG {

int Parameters::min_descriptor_height()
{
	assert(models.size() > 0 );
	int r = INT_MAX;
	for(size_t ii=0; ii < models.size(); ii++)
		r = std::min(models[ii].HOG_DESCRIPTOR_HEIGHT, r);
	return r;
}

int Parameters::min_descriptor_width()
{
	assert(models.size() > 0 );
	int r = INT_MAX;
	for(size_t ii=0; ii < models.size(); ii++)
		r = std::min(models[ii].HOG_DESCRIPTOR_WIDTH, r);
	return r;
}

int Parameters::min_window_height()
{
	assert(models.size() > 0 );
	int r = INT_MAX;
	for(size_t ii=0; ii < models.size(); ii++)
		r = std::min(models[ii].HOG_WINDOW_HEIGHT, r);
	return r;
}

int Parameters::min_window_width()
{
	assert(models.size() > 0 );
	int r = INT_MAX;
	for(size_t ii=0; ii < models.size(); ii++)
		r = std::min(models[ii].HOG_WINDOW_WIDTH, r);
	return r;
}

int Parameters::max_window_height()
{
	assert(models.size() > 0 );
	int r = 0;
	for(size_t ii=0; ii < models.size(); ii++)
		r = std::max(models[ii].HOG_WINDOW_HEIGHT, r);
	return r;
}

int Parameters::max_window_width()
{
	assert(models.size() > 0 );
	int r = 0;
	for(size_t ii=0; ii < models.size(); ii++)
		r = std::max(models[ii].HOG_WINDOW_WIDTH, r);
	return r;
}

void trim(std::string& s)
{
	using namespace std;
	if(s.empty()) return;

	int start = s.find_first_not_of(" \t");
	int end = s.find_last_not_of(" \t");
	string tmp = s.substr(start, end - start + 1);
	s.erase();
	s = tmp;
}

void extract_path(std::string& s, std::string& path)
{
	int last_slash = s.rfind('/');

	if( last_slash <= 0 ) {
		path = s;
	} else {
		path = s.substr(0,last_slash);
	}

}

int Parameters::load_from_file(std::string& fn)
{
	printf("reading Parameters from file: %s\n", fn.c_str());

	// save the path to the config file (the model files are there, too)
	extract_path(fn, path);

	using namespace std;
	ifstream is(fn.c_str());
	if(!is.good()) {
		printf("failed to open parameters file: %s\n", fn.c_str());
		return -1;
	}

	char c;

	while(is.good()) {
		is.read(&c, 1);
		char section_name[128];
		char comment[256];

		if( c == '[' ) {	// new section
			is.getline(section_name, 128, ']');

			// create a new ModelParameters object for the parameters to come
			ModelParameters m;
			m.identifier = string(section_name);

			printf("new ModelParameters object: %s\n", m.identifier.c_str());

			models.push_back(m);
		}
		else if( c == '#' ) {	// comment
			is.getline(comment, 256);
//			printf("ignoring comment line: %s\n", comment);
		} else {
			is.putback(c);
			// parse a key-value pair
			if( !strnlen(section_name, 128) || !strncmp(section_name, "global", 128) ) {
				// parse the general section
				string line;
				getline(is, line);

				istringstream iss(line);
				// we are in a models section
				char tmp[128];
				iss.getline(tmp, 128, '=');
				string key(tmp); trim(key);
				iss.getline(tmp,128);
				string value(tmp); trim(value);

				if( key.empty() ) continue;	// empty line
				if( value.empty() ) {
					printf("parser error: empty value for key: %s\n", key.c_str());
					return -1;
				}

				printf("\n\n\t\t!!! WARNING !!!\n");
				printf("currently global parameters have no effect!\n");

			} else {
				// parse a model section
				string line;
				getline(is, line);
				istringstream iss(line);
				char tmp[128];
				iss.getline(tmp, 128, '=');
				string key(tmp); trim(key);

				iss.getline(tmp,128);
				string value(tmp); trim(value);

				if( key.empty() ) continue;	// empty line
				if( value.empty() ) {
					printf("parser error: empty value for key: %s\n", key.c_str());
					return -1;
				}

				assert(models.size() > 0 );
				ModelParameters& model = models[models.size()-1];
				if( !key.compare("HOG_DESCRIPTOR_HEIGHT") ) {
					model.HOG_DESCRIPTOR_HEIGHT = atoi(value.c_str());
				} else if( !key.compare("HOG_DESCRIPTOR_WIDTH") ) {
					model.HOG_DESCRIPTOR_WIDTH = atoi(value.c_str());
				} else if( !key.compare("HOG_WINDOW_HEIGHT") ) {
					model.HOG_WINDOW_HEIGHT = atoi(value.c_str());
				} else if( !key.compare("HOG_WINDOW_WIDTH") ) {
					model.HOG_WINDOW_WIDTH = atoi(value.c_str());
				} else if( !key.compare("FILE") ) {
					model.filename = value;
				} else if( !key.compare("MIN_SCALE") ) {
					model.min_scale = atof(value.c_str());
				} else if( !key.compare("MAX_SCALE") ) {
					model.max_scale = atof(value.c_str());
				} else {
					printf("\nsection: %s\n", section_name);
					printf("unknown key value pair\t");
					printf("key:\t %s\t=\t%s\n\n", key.c_str(), value.c_str());
					return -2;
				}
			}
		}
	}
	return 0;
}

int ModelParameters::dimension()
{
	return HOG_DESCRIPTOR_HEIGHT * HOG_DESCRIPTOR_WIDTH * NBINS * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y;
}

}	// end of namespace cudaHOG
