# CUDA HOG ROS

![cuda_hog_ros_image](https://github.com/warp1337/cudahog_ros/blob/master/cuda_hog.png)

This **ROS** package is, basically, a 'modernized' and streamlined version
of the famous groundHOG component developed in the SPENCER project:
https://github.com/spencer-project/spencer_people_tracking

This 'hard fork' contains an updated/patched version of 
libcudaHOG which has been modified in order to work on recent Ubuntu 
(Debian) versions. Moreover, cudaHOG (groundHOG, respectively) has 
been *decoupled* from the SPENCER eco-system to run stand-alone. Yay!.
Lastly, the installation procedure of this project contains all steps 
and modifications that had to be done manually earlier, e.g., 
setting the required/target CUDA architecture.

## Installation

First, install CUDA. Either use the stand-alone installer or the
package management of your choice, e.g, apt. In my experience it 
is a good idea to use the stand-alone cuda installer since you can 
choose your install destination. However, by default CUDA is
installed in /usr/local/lib64

Install OpenCV. Again, choose your weapon, either install from source
or use apt. Compiling OpenCV yourself has the benefit of optimized
binaries, if you choose the corresponding CXXFLAGS, for instance. 
However, using apt is also fine.

Since this package is named cudahog_**ROS**, it is assumed that you already 
have ROS installed. Okay, now clone this repo!

First, build libcudaHOG. You need to set **CUDA_HOME**! Usually, if you
install CUDA by using apt it is located in /usr/local/cuda. If you
installed CUDA using the custom installer and you changed the location
you will most probably know where you cuda installation is located.

<pre>
cd cudahog_ros && cd cudahog_lib
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=***YOUR_CHOICE*** -DCUDA_HOME=***WHERE_CUDA_IS_INSTALLED***
make && make install
</pre>

An example assuming you installed CUDA in $HOME/test_system and you also want
cudahog_ros installed in this folder/prefix. Note: the two paths do **not** necessarily
be the same. This is just an example.

<pre>
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/test_system -DCUDA_HOME=$HOME/test_system
</pre>

Now build cudahog_ros

<pre>
cd cudahog_ros && cd cudahog_ros
source /opt/ros/kinetic.setup.bash
alterntive: use your catkin_ws and source devel/setup.bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=***YOUR_CHOICE***
make && make install
</pre>

You are done.

## Usage

<pre>
source $INSTALL_PREFIX/setup.bash
alterntive: source devel/setup.bash
roslaunch cudahog_ros cudahog.launch
</pre>

You will most probably need to change the default image topic
in the cudahog.launch

<pre>
/pepper_robot/sink/front

Here, image_raw is appended in the code, so just change the prefix...
</pre>

## Further Plans

Currently, we are training a new model, based on libcudaHOG that
detects upper-bodies only, in contrast the the current model that
detects pedestrians (needs legs for instance).

## Credits
 
 - groundHOG has been provided by RWTH Aachen: 
 - http://www.vision.rwth-aachen.de/projects/
 - The initial ROSification was done by the SPENCER people: 
 - https://github.com/spencer-project/spencer_people_tracking/tree/master/detection/monocular_detectors/rwth_ground_hog
 - This package has been created and modified by Florian Lier: 
 - https://cit-ec.de/en/central-lab-facilities/contacts