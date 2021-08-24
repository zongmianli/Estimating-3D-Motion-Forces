# Estimating 3D Motion & Forces of Person-Object Interactions from Monocular Video

Zongmian Li, Jiri Sedlar, Justin Carpentier, Ivan Laptev, Nicolas Mansard and Josef Sivic

[Project page](https://www.di.ens.fr/willow/research/motionforcesfromvideo/)

## Introduction

The 3D motion-force estimator relies on a number of open-source libraries listed as follows.
As a quick setup, it is recommended to compile the estimation stage and test the demo code with the provided recognition stage outputs.
Otherwise the recognition stage is required if you want to test the method on new videos.

## Recognition stage

The recognition stage is to be installed and run first on the input video.
It consists of the following sub-projects:

- Openpose (we are using this [fork](https://github.com/zongmianli/Realtime_Multi-Person_Pose_Estimation))
- HMR (use this [fork](https://github.com/zongmianli/HMR-imagefolder) instead)
- [Contact recognizer](https://github.com/zongmianli/contact-recognizer)
- [Object endpoint recognizer](https://github.com/sedlaj45/endpoints)

## Estimation stage

### Dependencies

- Python 2 (tested on version 2.7.15)
- [CMake](https://cmake.org/) (tested on version 3.13.4)
- [Boost](https://www.boost.org/) with component Python (tested on version 1.68.0)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (tested on version 3.3.7)
- [EigenPy](https://github.com/stack-of-tasks/eigenpy) (required for the python bindings; tested on version 1.5.0.)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) with option Python binding (tested on version 2.0.0)
- [Ceres solver](http://ceres-solver.org/installation.html) with components EigenSparse, SparseLinearAlgebraLibrary, LAPACK, SuiteSparse, CXSparse, AccelerateSparse, SchurSpecializations (tested on version 2.0.0)
- [Googletest](https://github.com/google/googletest) (for building unit tests)
- [virtualenv](https://virtualenv.pypa.io/en/latest/) (for creating isolated Python environments)

### Installation

Please note that, for convenience, we will use `~/Estimating-3D-Motion-Forces` as the default local directory for hosting the code base and the demo data.

1. Make a local copy of the project:
   ```terminal
   git clone https://github.com/zongmianli/Estimating-3D-Motion-Forces.git ~/Estimating-3D-Motion-Forces
   cd ~/Estimating-3D-Motion-Forces
   ```
   
2. Move to `~/Estimating-3D-Motion-Forces`, create a top-level directory `build` and move there:
   ```terminal
   cd ~/Estimating-3D-Motion-Forces
   mkdir build && cd build
   ```
   
3. Create an isolated Python enviroment for the project:
   ```terminal
   virtualenv venv_3dmf_est -p python2.7
   source venv_3dmf_est/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
   
4. Configure with CMake in *Release* mode:
   ```terminal
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
   Make sure that CMake can find all the dependencies with correct version.

5. Build the project from source code:
   ```terminal
   make -j4
   ```

### Demo

1. Download and set up the [Handtool dataset](https://github.com/zongmianli/Handtool-dataset) in a desired directory, for example, `~/Handtool-dataset`.

2. Download pre-computed 2D measurements.
   The downloaded data include the person 2D poses, object 2D endpoints, contact states and person intial motion trajectories pre-computed on the Handtool dataset.
   ```terminal
   source scripts/setup_precomputed_data.sh ~/Handtool-dataset
   ```
   
2. Update the video name (`hammer_1` by default) and the parameters in the begining of [scripts/run_video.sh](tbd).
   Run the following script to run the 3D motion-force estimator:
   ```terminal
   source scripts/run_video.sh
   ```
   The estimated 3D motion and contact forces are saved in `results/` by default.

## Citation

If you find the code useful, please consider citing:
```bibtex
@InProceedings{li2019motionforcesfromvideo,
  author={Zongmian Li and Jiri Sedlar and Justin Carpentier and Ivan Laptev and Nicolas Mansard and Josef Sivic},
  title={Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
