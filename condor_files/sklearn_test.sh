#!/bin/bash

# untar your Python installation
tar -xzf combined.tar.gz
# make sure the script will use your Python installation, 
# and the working directory as it's home location
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home
# run your script
python genMGP_trainRF1.py
