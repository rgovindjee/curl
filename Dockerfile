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

# Install Python 3.9 and make default python3
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN sudo apt-get update
RUN sudo apt install -y python3.9
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
# Install pip correctly so pip3 aliases
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3.9 get-pip.py
RUN sudo apt-get install -y python3-pip python3.9-distutils
RUN pip3 install --upgrade pip  # May need to update pip to install opencv
RUN pip3 install setuptools==46.0.0 # Packages below require this
RUN pip3 install gym[atari,accept-rom-license] # Replace with gymnasium?

# Install other necessities
RUN sudo apt-get install -y vim

# Install python packages
RUN pip3 install torch numpy
# Use < version 4.3 to avoid having to rebuild opencv C++ from source.
RUN pip3 install "opencv-python<4.3" -v # Chnage to opencv-contrib-python if needed.
RUN pip3 install gym[atari,accept-rom-license] # seemingly not compatible with python3.7 and associated packages. Replace with gymnasium?
RUN pip3 install gym_super_mario_bros
RUN pip3 install Pillow # For super mario with gymnasium
RUN pip3 install tensorboardX # For super mario with gymnasium
RUN pip3 install gymnasium[atari,accept-rom-licese,classic-control] #for basic usage cartpole

