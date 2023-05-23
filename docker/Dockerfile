FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

MAINTAINER Luca Di Giammarino <digiammarino@diag.uniroma1.it>

ARG STANDARD_PKG="git vim wget"

ARG LIBRARIES="sudo libeigen3-dev \
 libsuitesparse-dev \
 libqglviewer-dev-qt5 \
 freeglut3-dev \
 libpcl-dev"

# installing all the packages
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y ${STANDARD_PKG} ${LIBRARIES}

# installing ROS
RUN /bin/sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' 
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && apt-get install ros-noetic-desktop-full -y
RUN apt-get install -y python3-pip python3-catkin-tools ros-noetic-grid-map-msgs
RUN pip3 install numpy 

# create workspace
RUN mkdir -p /catkin_ws/src

# clone required pkgs
RUN cd /catkin_ws/src && git clone https://gitlab.com/srrg-software/srrg_cmake_modules.git   
RUN cd /catkin_ws/src && git clone https://gitlab.com/srrg-software/srrg_hbst.git 
RUN cd /catkin_ws/src && git clone https://gitlab.com/srrg-software/srrg2_core.git && cd srrg2_core && git checkout 9cc15007 
RUN cd /catkin_ws/src && git clone https://gitlab.com/srrg-software/srrg2_solver.git && cd srrg2_solver && git checkout 4eb02cf8 

# remove useless stuff from solver
RUN cd /catkin_ws/src/srrg2_solver && rm -rf srrg2_solver_gui srrg2_solver_star srrg2_solver_experiments srrg2_solver_calib_addons srrg2_solver_extras
RUN . /opt/ros/noetic/setup.sh && cd /catkin_ws && catkin init && catkin build -DSRRG_ENABLE_CUDA=ON

# run md slam without building 
RUN cd /catkin_ws/src && git clone https://github.com/digiamm/ba_md_slam.git

# enable gl viewer
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*
  
# env vars for the nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# dynamic libraries loading srrg stuff
COPY ./dl.conf /dl.conf

WORKDIR /
CMD ["bash"]

# add some aliases and rospath to bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "alias ws2='source devel/setup.bash'" >> ~/.bashrc
RUN echo "alias gpu_build='catkin build -DSRRG_ENABLE_CUDA=ON && ws2'" >> ~/.bashrc




