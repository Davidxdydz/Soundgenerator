# Sound generator project

We make sounds using cool math and machine learning.

---

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
$ sudo apt install python3 python3-dev python3-pip build-essential tqdm
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
$ python3 main.py
```
### Arch
```shell
$ sudo pacman -S python3 python3-pip base-devel
$ pip3 install numpy matplotlib tensorflow scipy tqdm
$ sudo pip3 install "pybind11[global]"
$ git clone https://github.com/Davidxdydz/Soundgenerator.git
$ cd Soundgenerator
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ mv *.so ..
$ cd ..
$ python3 main.py
```

### Windows
Additional prerequesite: msbuild  
works with msbuild Version 16.1.76+g14b0a930a7, other versions probably work as well.  
Msbuild is not in PATH as standard, use it e.g. from "Developer Command Prompt for VS xx"  
Add pybind to PATH

```
$ pip install pybind11[global] numpy matplotlib tensorflow scipy tqdm
$ git clone https://github.com/Davidxdydz/Soundgenerator.git
$ cd Soundgenerator
$ mkdir build
$ cmake ..
$ msbuild functionGenerator.sln
$ move .\Debug\function_generator.cp38-win_amd64.pyd ..
$ cd ..
$ python main.py
```
The build steps are automated in build_windows.py, if PATH is set up correctly.  
Execute in `Soundgenerator/` with `python build_windows.py`

## Troubleshooting
No problems encountered yet! :)

## Documentation
Probably once we set up doxygen or something...

## Contribute
Yes.

## License
None, currently.


