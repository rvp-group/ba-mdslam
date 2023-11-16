<h1>Installation</h1>

## Install [CUDA](https://developer.nvidia.com/cuda-downloads) (tested versions 8, 9.1, 10.1)

## Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on Ubuntu 20.04


Once CUDA and ROS are installed, run
``` 
sudo apt-get update 
```
Now install the required extra packages
``` 
sudo apt-get install libeigen3-dev libsuitesparse-dev libqglviewer-dev-qt5 freeglut3-dev libpcl-dev ros-noetic-grid-map-msgs python3-catkin-tools
```
Create a folder for the ROS workspace and go into it
```
mkdir -p /catkin_ws/src && cd /catkin_ws/src 
```
Clone this package and other dependencies on the `src` folder
```
cd ~/catkin_ws/src/
git clone https://github.com/digiamm/ba_md_slam.git
git clone https://gitlab.com/srrg-software/srrg_cmake_modules.git 
git clone https://gitlab.com/srrg-software/srrg_hbst.git 
git clone https://gitlab.com/srrg-software/srrg2_core.git && cd srrg2_core 
git clone https://gitlab.com/srrg-software/srrg2_solver.git && cd srrg2_solver 

```
Checkout `srrg2_core` and `srrg2_solver` to tested version
```
git checkout ~/catkin_ws/src/srrg2_core 9cc15007
git checkout ~/catkin_ws/src/srrg2_solver 4eb02cf8
```
Remove useless stuff we don't need from solver for easier building
```
cd /catkin_ws/src/srrg2_solver && rm -rf srrg2_solver_gui srrg2_solver_star srrg2_solver_experiments srrg2_solver_calib_addons srrg2_solver_extras
```

Build package and dependencies using `catkin_tools`
```
cd ~/catkin_ws && catkin build -DSRRG_ENABLE_CUDA=ON
```
Finally, source workspace
```
source ~/catkin_ws/devel/setup.bash
```
