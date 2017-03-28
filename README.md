# libfreenect2_altered

## Description
Altered driver for Kinect for Windows v2 and Xbox One devices so that amplitude and phase of each frequency are extracted.

## Requirements
Requirements are the same as the [original libfreenect2](https://github.com/OpenKinect/libfreenect2). Please confirm that the build and run propery work on your computer using the original version. 

Note: Source codes in the 'master' branch are not modified except for the README.

## Extract amplitude and phase of each frequency

#### Quick start
Firstly, checkout the branch `for_static_scene` and build as:
```
git checkout for_static_scene
mkdir build && cd build
cmake ..
make
```
Amplitude and phase data are extracted by CPU processing mode:
```
bin/Protonect cpu
```
Amplitude, phase, and RGB images are stored as binary files. 
Please note that these images are averaged over the all frames, so the scene should be static.
To gain good SNR, a few hundreds of frames should be obtained.

To convert binary file to images, `tools/dat2png.py` will work.

Note: our environment was Ubuntu Desktop 14.04.3 and python 2.7.6.

#### Other branches
We provide other branches below. Please note that souce codes in these branches are not organized and README is not provided yet. We cannot provide any support regarding these branches.
* `demo`:
 The real time demo app. Database is also provided, but it is device-dependent. The accuracy is not guaranteed.
* `stage_sync`:
 The newest source codes to capture distortions using OptoSigma translation stages.

