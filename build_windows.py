# prerequesites:
#   cmake in PATH
#   msbuild in PATH
#   pybind in PATH

# call in Soundgenerator/
# python build_windows.py

# works only for debug builds

import subprocess
import os
import shutil

if not os.path.exists("build"):
    os.mkdir("build")

os.chdir("build")
subprocess.call(["cmake",".."])
subprocess.call(["msbuild","functionGenerator.sln"])
os.chdir("..")

print()
for file in os.listdir("build/Debug"):
    if file.endswith(".pyd"):
        path = os.path.join("build/Debug",file)
        print("copy",path)
        shutil.copy(path,file)