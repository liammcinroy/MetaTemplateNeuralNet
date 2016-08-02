#!/bin/bash

cp Debug/CNN.lib CNN/imatrix.h CNN/ilayer.h CNN/neuralnet.h CNN/neuralnetanalyzer.h Builds/Debug/
cp Release/CNN.lib CNN/imatrix.h CNN/ilayer.h CNN/neuralnet.h CNN/neuralnetanalyzer.h Builds/Release/

rm -r Debug/
rm -r Release/
