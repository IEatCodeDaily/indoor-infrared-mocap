import sysconfig
from distutils.sysconfig import get_python_lib, get_python_inc
import os
import json
import numpy

def get_python_dirs():
    python_info = {}
    libdir = sysconfig.get_config_var('LIBDIR')
    if libdir is None:
        libdest = sysconfig.get_config_var('LIBDEST')
        if libdest is not None:
            libdir = os.path.join(os.path.dirname(libdest), 'libs')

    python_info["Version"] = sysconfig.get_python_version().replace('.', '')
    python_info["LibDir"] = libdir.replace('\\', '/')
    python_info["IncludeDir"] = get_python_inc().replace('\\', '/')
    python_info["PackagesPath"] = get_python_lib().replace('\\', '/')
    python_info["NumPyInclude"] = numpy.get_include().replace('\\', '/')
    
    
    return python_info

def main():
    python_info = get_python_dirs()
    
    with open("scripts/get_python_dir.json", "w") as f:
        json.dump(python_info, f, indent=4)
    
    # print("Using paths:")
    # for key, value in python_info.items():
    #     print(f"{key}: {value}")

if __name__ == "__main__":
    main()