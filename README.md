# Sound generator project

We make sounds using cool math and machine learning.

## Features

Not many currently...

## Prerequisites

* Python3.5 and higher
* CMake 3.10 and higher
* A C/C++ compiler that supports C++11 and higher

## Build

This project relies on C++ code for training data generation, you will have to build this
yourself. Here is a short guide on how to build on different platforms.

### Debian and Ubuntu
```shell
$ sudo apt install python3 python3-dev python3-pip build-essential
$ pip3 install numpy matplotlib tensorflow scipy
$ sudo pip3 install "pybind11[global]"
$ git clone https://github.com/Davidxdydz/Soundgenerator.git
$ cd Soundgenerator
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ mv *.so ..
$ cd ..
$ python3 createSounds.py
```
### Arch
```shell
$ sudo pacman -S python3 python3-pip base-devel
$ pip3 install numpy matplotlib tensorflow scipy
$ sudo pip3 install "pybind11[global]"
$ git clone https://github.com/Davidxdydz/Soundgenerator.git
$ cd Soundgenerator
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ mv *.so ..
$ cd ..
$ python3 createSounds.py
```

### Windows
Good luck!

## Troubleshooting
No problems encountered yet! :)

## Documentation
Probably once we set up doxygen or something...

## Contribute
Yes.

## License
None, currently.


