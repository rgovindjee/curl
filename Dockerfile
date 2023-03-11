# VNC image with ROS Melodic
FROM fbottarel/ros-desktop-full-vnc

# Install Gazebo
#RUN sudo apt-get install ros-melodic-ros-gz

RUN sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
RUN sudo apt-get update
RUN sudo apt-get install -y gazebo9
RUN sudo apt-get install -y libgazebo9-dev
RUN sudo apt-get install -y ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y libignition-math2
