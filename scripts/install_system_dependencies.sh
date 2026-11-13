#!/usr/bin/env bash
# Install the Ubuntu/ROS components used by the simulation demo.
set -euo pipefail

source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
  echo 'This installer requires Ubuntu 24.04 and ROS 2 Jazzy.' >&2
  exit 1
fi
if [[ ! -f /etc/apt/sources.list.d/ros2.sources && ! -f /etc/apt/sources.list.d/ros2.list ]]; then
  if ! dpkg-query -W -f='${Status}' ros2-apt-source 2>/dev/null | grep -q 'install ok installed'; then
    echo 'Install the official ros2-apt-source package first; see the README.' >&2
    exit 1
  fi
fi

packages=(
  ros-jazzy-ur-simulation-gz
  ros-jazzy-ur-moveit-config
  ros-jazzy-ros-gz-bridge
  ros-jazzy-ros-gz-image
  ros-jazzy-cv-bridge
  python3-opencv
  python3-pil
  python3-pytest
  python3-colcon-common-extensions
)

sudo apt-get update
sudo apt-get install --no-install-recommends --no-remove -y "${packages[@]}"
/usr/bin/python3 -c 'import cv2; assert hasattr(cv2, "aruco"), "OpenCV ArUco module is required"; print("OpenCV:", cv2.__version__)'
echo 'System dependencies installed. Return to the implementation session.'
