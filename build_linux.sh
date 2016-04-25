#!/bin/sh
set -e

# Simple script to build a mostly static version of cgminer for decred
# for linux.
# This could definitely be smarter and not always rebuild the deps.
# You must already have downloaded the ADL sdk from AMD and put the
# zip file in the same directory as the cgminer repo to use this.
#
# 2016/04/25
# jcv@decred.org

touch build.log

echo "Downloading dependencies if needed."
mkdir -p ../cgminer-static
cd ../cgminer-static/
ZLIB=zlib-1.2.8
if [ ! -e $ZLIB.tar.gz ]
then
    wget -q http://zlib.net/$ZLIB.tar.gz
fi
CURL=curl-7.48.0
if [ ! -e $CURL.tar.gz ]
then
    wget -q https://curl.haxx.se/download/$CURL.tar.gz
fi
NCURSES=ncurses-6.0
if [ ! -e $NCURSES.tar.gz ]
then
    wget -q https://ftp.gnu.org/gnu/ncurses/$NCURSES.tar.gz
fi

PREF=$PWD
cd $PREF

rm -rf bin/ $CURL include/ lib/ $NCURSES share/ $ZLIB

echo "Building zlib."
tar -xzf $ZLIB.tar.gz
cd $ZLIB
./configure --static --prefix=$PREF >> build.log 2>&1
make install >> build.log 2>&1
cd ..

echo "Building curl."
tar -xzf $CURL.tar.gz
cd $CURL
./configure --disable-rt --disable-rtsp --disable-libcurl-option --disable-ipv6 --disable-verbose --without-ca-bundle --disable-telnet --disable-tftp --disable-pop3 --disable-imap --disable-smb --disable-smtp --disable-gopher --disable-ftp --disable-file --disable-dict --disable-unix-sockets --disable-manual --disable-shared --enable-static --prefix=$PREF >> build.log 2>&1
make install >> build.log 2>&1
cd ..

echo "Building ncurses."
tar -xzf $NCURSES.tar.gz
cd $NCURSES
CC=clang Cxx=clang++ ./configure  --enable-pc-files --enable-termcap --without-shared --enable-static --prefix=$PREF --with-pkg-config-libdir=$PREF/lib/pkgconfig --with-terminfo-dirs=/lib/terminfo:/usr/share/terminfo >> build.log 2>&1
make install >> build.log 2>&1
cd ..

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
unzip -q $ADL.zip
cp include/* .

echo "Building cgminer."
cd ..
PKG_CONFIG_LIBDIR=$PREF/lib/pkgconfig/ ./autogen.sh --enable-opencl >> build.log 2>&1
make >> build.log 2>&1
