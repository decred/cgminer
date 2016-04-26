#!/bin/sh
set -e

# Simple script to build a mostly static version of cgminer for decred
# for windows.
# This could definitely be smarter.
# You must already have downloaded the ADL sdk from AMD and put the
# zip file in the same directory as the cgminer repo to use this.
# You also need a proper x86_64-w64-mingw32 toolchain and
# mxe's dependencies installed (see http://mxe.cc for more info).
#
# Download the AMD APP SDK from:
#
# http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
#
# And create an AMD_OpenCL.zip file like so:
# 
#$ unzip -l AMD_OpenCL.zip 
#Archive:  AMD_OpenCL.zip
#  Length      Date    Time    Name
#---------  ---------- -----   ----
#        0  2016-04-26 14:38   include/
#        0  2016-04-26 14:37   include/CL/
#    43667  2016-04-26 11:22   include/CL/cl_platform.h
#    72358  2016-04-26 11:22   include/CL/cl.h
#     5308  2016-04-26 11:22   include/CL/cl_dx9_media_sharing.h
#     5003  2016-04-26 11:22   include/CL/cl_d3d11.h
#     5009  2016-04-26 11:22   include/CL/cl_d3d10.h
#     7508  2016-04-26 11:22   include/CL/cl_gl.h
#     1808  2016-04-26 11:22   include/CL/opencl.h
#   289468  2016-04-26 11:22   include/CL/cl.hpp
#    19917  2016-04-26 11:22   include/CL/cl_ext.h
#     2699  2016-04-26 11:22   include/CL/cl_gl_ext.h
#        0  2016-04-26 14:37   lib/
#    25870  2016-04-26 11:21   lib/OpenCL.lib
#    85766  2016-04-26 11:21   lib/libOpenCL.a
#---------                     -------
#   564381                     15 files
#
#
# 2016/04/26
# jolan@decred.org

echo "Building dependencies via mxe."
mkdir -p ../cgminer-static
cd ../cgminer-static
if [ ! -e mxe ]
then
	git clone https://github.com/mxe/mxe.git
fi
cd mxe
make MXE_TARGETS='x86_64-w64-mingw32.static' curl ncurses pthreads
cd ..

# OpenCL
echo "Getting OpenCL headers and library."
cd ..
OPENCL=AMD_OpenCL.zip
if [ ! -e $OPENCL ]
then
	echo "OpenCL headers and library not found in $PWD.  See script comments for more information."
	exit
fi
cp $OPENCL cgminer-static
cd cgminer-static
unzip -qo $OPENCL

# ADL
echo "Getting ADL headers."
cd ..
ADL=ADL_SDK9.zip
if [ ! -e $ADL ]
then
    echo "ADL headers not found in $PWD.  Download ADL_SDK9.zip from http://developer.amd.com/tools-and-sdks/graphics-development/display-library-adl-sdk/"
    exit
fi
cp $ADL cgminer/ADL_SDK/
cd cgminer/ADL_SDK/
unzip -qo $ADL
cp -f include/* .

echo "Building cgminer."
cd ..
env CFLAGS="-DCURL_STATICLIB" CPPFLAGS="-I`pwd`/../cgminer-static/include -I`pwd`/../cgminer-static/mxe/usr/x86_64-w64-mingw32.static/include" LDFLAGS="-L`pwd`/../cgminer-static/lib -L`pwd`/../cgminer-static/mxe/usr/x86_64-w64-mingw32.static/lib" sh autogen.sh --enable-opencl --prefix=/home/jolan/mxe/usr/bin --host=x86_64-w64-mingw32 --build=x86_64-linux
make V=1
