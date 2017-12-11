-------------------------------------------------------------------------------
-  groundHOG -
-------------------------------------------------------------------------------

This is a fast GPU-based (Histograms of Oriented Gradients) HOG detector implementation.
It can detect objects (like pedestrians or cars) in images.

It is part of the work presented in

Efficient Use of Geometric Constraints for Sliding-Window Object Detection in Video
P. Sudowe, B. Leibe
International Conference on Vision Systems (ICVS'11), 2011

and

Real-Time Multi-Person Tracking with Time-Constrained Detection,
D. Mitzel, P. Sudowe, B. Leibe.
British Machine Vision Conference (BMVC'11), 2011.


This release includes the features described in the two above papers, which allow
to make more efficient use of the computational resources. But the detector can
also be run exhaustively on the whole image. The detector is very fast and the
recognition performance is very competitive when compared to other published
implementations.
The original description of the HOG (Histograms of Oriented Gradients) detector
is due to N. Dalal and B. Triggs (CVPR 05).

-----------------------------------
CHANGELOG
-----------------------------------

cudaHOGDump:
	In version 1.0 the cudaHOGDump binary did not properly set the default value of the -x
	parameter. This affected the included train.sh example script. Handling of -x has
	been fixed. Note, that the detection/test phase was not affected. 

Binary Model Files:
	The svmdense implementation uses 'long', which translates to different bit-widths depending
	on the platform. We now always use int64_t. Old models trained on 32-bit systems should be
	retrained or manually translated.

Windows Support:
	Thanks to Javier Martin Tur, it is possible to compile and use groundHOG on Windows.

-----------------------------------
DESCRIPTION
-----------------------------------

groundHOG is a library which you can use in your own projects. The only header
cudaHOG.h forms the interface. Some compile-time parameters can be found in the
global.h, which influence the behaviour of the detector.

Distributed with the library is the following:

	* cudaHOG library - GPU-based HOG detector
	* cudaHOGDetect - command line tool to run detector on images
	* cudaHOGDump   - command line tool to dump features for training a new SVM
	* scripts		- to visualize the detection results & and example training script
	* model 		- this is a model for pedestrian detection trained
	                  on the INRIAPerson dataset.

-----------------------------------
COPYING  - LICENCE
-----------------------------------

If you use this software for research purposes we ask you to cite our paper:
BibTex:

@InProceedings{Sudowe11ICVS,
  author =       {P. Sudowe and B. Leibe},
  title =        {{Efficient Use of Geometric Constraints for Sliding-Window Object Detection in Video}},
  booktitle =    ICVS,
  OPTpages =     {},
  year =         {2011},
}

The software is released under the GPL v3. A copy of the licence can be found
in the COPYING.txt that is included with this software.
The pedestrian model is released under the same licence terms as the remaining software.

THIS CODE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE. Use at your own risk.

-----------------------------------
Dependencies
-----------------------------------

* NVIDIA CUDA enabled GPU in your system
* CUDA installed
* Qt 		- which is used to load images
* qmake 	- build tool

-----------------------------------
BUILD
-----------------------------------

Run this in the projects top-level directory:

$ qmake
$ make

This should build the library, cudaHOG, as well as the two binary
command line tools. If you are missing any of the above dependencies
you have to install them. Possibly you have to adapt some paths
in the cudaHOG.pro project files (in ./cudaHOG/), if qmake exits with errors.

-----------------------------------
USAGE  - Detection
-----------------------------------

Example:

	cudaHOGDetect --config model/config --directory ${DIRECTORY_OF_IMAGES} --output /tmp

	You will find for each image a .detections file in /tmp afterwards. The results can be
	visualized using the script in the scripts/ sub-folder.

Alternatively:

	cudaHOGDetect --config model/config --image ${IMAGE_FILE}

	This will run on just one single image, and not write the result out to a file, but
	only print the detection result to the terminal window.


--help will provide you with an overview of possible options.

Two important ones are:

-s  -- start scale (default: 1.0), 0.5 means to upscale the image by a factor of 2,
which enables one to find also the smaller objects

-f -- scale factor (default: 1.05), the scale step - depending on the application
it might be beneficial to increase this slightly, but beware this can advertly affect
detection performance.


-----------------------------------
USAGE  - Training a new model
-----------------------------------

There is an example script train_inria.sh which trains a new model on the INRIAPerson
model. It should be easy to adapt this to any other dataset. The model parameters
should be set in a config file.

The cudaHOGDump tool is used to dump features. The svmdense package is used to train
a new model. This can be downloaded from http://pascal.inrialpes.fr/soft/olt/ .
The library supports the binary models generated by this package only, but it would
be possible to adapt this to other model formats quite easily. For this please check
the code.

-----------------------------------
SUPPORT
-----------------------------------

This software is developed on Linux (Ubuntu 64Bit).

If there are questions regarding the implementation you may contact
Patrick Sudowe <sudowe@umic.rwth-aachen.de>.

If you should find any bugs or want to suggest improvements you are
most welcome to contact us. If you want to submit any patches, we
prefer git patch format.

-----------------------------------
CONTACT
-----------------------------------
Patrick Sudowe <sudowe@umic.rwth-aachen.de>

Downloaded from www.mmp.rwth-aachen.de/projects/groundhog
