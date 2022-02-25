#!/bin/sh

# Steps to set up new Lenovo machine with Atari Logger. Follow these before running script
# 1. Connect to WiFi
# 2. Download wsl2 using the manual installation steps from the Microsoft website
# 3. Download Xlaunch. Open it with all the defaults except check the box saying `No Access Control`. Save the config. Approve the public/private connections with Windows Defender
# 4. Turn off all Windows malware, weather updates, etc.
# 5. Clone git repo (you must have done this already to be reading this though...)

# set up Python
sudo apt-get update
sudo apt-get install libpython3-dev
sudo apt-get install python3-venv
python3.8 -m venv venv
source venv/bin/activate
cd atari_logger
pip install -r requirements.txt

# Download GNU Multiple Precision Library for envlogger
sudo apt-get install libgmp3-dev

# set up logging folder
cd
mkdir log

# set up bashrc by: 1) enabling X11 Forwarding, 2) saving Windows Username as a bash variable
cat <<EOT >> ~/.bashrc
# X11 Forwarding
export DISPLAY=$(ip route list default | awk '{print $3}'):0
export LIBGL_ALWAYS_INDIRECT=1
export LD_LIBRARY_PATH=/usr/local/lib

# Saving Windows username to be used in python script
export USERNAME=`cmd.exe /c echo %username%`
EOT

# set up atari_logger configuration file
source ~/.bashrc
cat <<EOT > /mnt/c/Users/$USERNAME/Desktop/game_config.txt
1
montezuma_revenge
EOT
