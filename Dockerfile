# VNC image with ROS Melodic
FROM fbottarel/ros-desktop-full-vnc

# Install Gazebo
RUN sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
RUN sudo apt-get update
RUN sudo apt-get install -y gazebo9
RUN sudo apt-get install -y libgazebo9-dev
RUN sudo apt-get install -y ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y libignition-math2

# Install other necessities
RUN sudo apt-get install -y python3-pip vim

# Install python packages
RUN pip3 install torch numpy
RUN pip3 install --upgrade pip  # Need to update pip to install opencv
# Use < version 4.3 to avoid having to rebuild opencv C++ from source.
RUN pip3 install "opencv-python<4.3" -v # Chnage to opencv-contrib-python if needed.
RUN pip3 install gym-super-mario-bros Pillow # For super mario with gymnasium
RUN pip3 install tensorboardX # For super mario with gymnasium
RUN pip3 install gym[atari,accept-rom-license]==0.21.0
