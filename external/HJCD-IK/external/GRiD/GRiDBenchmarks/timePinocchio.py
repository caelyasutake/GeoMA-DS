# Running this code with floating base is still under development

#!/usr/bin/python3
import GRiD.util as util
import subprocess
import sys

def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE = util.parseInputs()
    util.validateFile(URDF_PATH)

     
    
    print("-----------------")
    print("Compiling timePinocchio")
    print("   this may take a few minutes")
    print("-----------------")
    
    # Run pkg-config to get the flags and libraries
    pkg_config_command = ['pkg-config', '--cflags', '--libs', 'pinocchio', 'cppadcg']
    result = pkg_config_result = subprocess.run(pkg_config_command, capture_output=True, text=True, check=True)

    if result.stderr:
        print("Compilation errors follow:")
        print(result.stderr)
        exit()

    # Construct the full compiler command
    compiler_command = [
        'g++', '-std=c++11', 'timePinocchio.cpp', '-o', 'timePinocchio.exe', '-O3'
    ] + pkg_config_result.stdout.strip().split()

    # Run clang++ with the constructed command
    result = subprocess.run(compiler_command, capture_output=True, text=True)

    if result.stderr:
        print("Compilation errors follow:")
        print(result.stderr)
        exit()

    print("-----------------")
    print("Running timePinocchio")
    print("-----------------")
    print("This may take a few minutes....")
    print("     Outputs will show up at the end")
    print("-----------------")
    result = subprocess.run(["./timePinocchio.exe", URDF_PATH, str(FLOATING_BASE)], capture_output=True, text=True)
    if result.stderr:
        print("Runtime errors follow:")
        print(result.stdout)
        print(result.stderr)
        exit()

    print(result.stdout)

if __name__ == "__main__":
    main()