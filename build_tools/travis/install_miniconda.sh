#!/bin/bash

# partly obtained from https://github.com/mdtraj/mdtraj/blob/master/devtools/travis-ci/install_miniconda.sh

if [[ "$TRAVIS_OS_NAME" == "osx" ]];   then MINICONDA=Miniconda3-latest-MacOSX-x86_64.sh; fi
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then MINICONDA=Miniconda3-latest-Linux-x86_64.sh;  fi

MINICONDA_MD5=$(curl -s https://repo.continuum.io/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
wget https://repo.continuum.io/miniconda/$MINICONDA
if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
    echo "Miniconda MD5 mismatch"
    exit 1
fi


if [[ ${TRAVIS_OS_NAME} = osx ]]; then
    # Since default gcc on osx is just a front-end for LLVM
    if [[ ${CC} = gcc ]]; then
        export CXX=g++-4.8
        export CC=gcc-4.8
    fi
fi


bash $MINICONDA -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

rm -f $MINICONDA
