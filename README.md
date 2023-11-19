# Estimating 3D Motion & Forces of Person-Object Interactions from Monocular Video

Zongmian Li, Jiri Sedlar, Justin Carpentier, Ivan Laptev, Nicolas Mansard and Josef Sivic

This repository hosts the code of the estimation stage for the [Motion-Forces-from-Video](https://www.di.ens.fr/willow/research/motionforcesfromvideo/) project.

![Parkour Demo](./demo-parkour.gif)

## Introduction

The 3D motion-force estimator relies on a number of open-source libraries listed as follows.
As a quick setup, users can compile the solver and test the demo code using precomputed data.

The recognition stage is only required if you want to apply the method to new videos. 
If this is the case, consider installing the following open-source projects:

- Openpose (we are using this [fork](https://github.com/zongmianli/Realtime_Multi-Person_Pose_Estimation)), or any perferred 2D pose estimator.
- HMR (use this [fork](https://github.com/zongmianli/HMR-imagefolder) instead), or any 3D pose estimator that outputs SMPL joint angles. Another good option is [FrankMocap](https://github.com/facebookresearch/frankmocap).
- Use [Contact recognizer](https://github.com/zongmianli/contact-recognizer), or the [hand-object detector from Michigan](https://github.com/ddshan/hand_object_detector).
- Use [Object endpoint recognizer](https://github.com/sedlaj45/endpoints), or any object segmentation model of your choice.

## Installing via pip/conda
We are working on this. Stay tuned!

## Build from source

### Dependencies

- Python 3 (tested on version 3.8.17)
- [CMake](https://cmake.org/) (tested on version 3.27.5)
- [Boost](https://www.boost.org/) with component Python (tested on version 1.78.0)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (tested on version 3.4.0)
- [EigenPy](https://github.com/stack-of-tasks/eigenpy) (required for the python bindings; tested on version 3.1.0.)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) with option Python binding (tested on version 2.6.20)
- [Ceres solver](http://ceres-solver.org/installation.html) with components EigenSparse, SparseLinearAlgebraLibrary, LAPACK, SuiteSparse, CXSparse, AccelerateSparse, SchurSpecializations (tested on version 2.2.0)

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
   
3. Install the dependencies in an isolated enviroment for the project. `conda` is recommended.
   
4. Configure with CMake in *Release* mode:
   ```terminal
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
   Make sure to add `-DCMAKE_BUILD_TYPE=Release` at this stage. Otherwise Ceres will run in debug mode and will be very slow.

5. Build the project from source code:
   ```terminal
   make -j4
   ```
The compilation generally takes around 2-10 minutes, depending on your hardware. 
Finally you will find the compiled solver library in `~/Estimating-3D-Motion-Forces/lib/solver.so`.
## Demo code
Run the following script to generate the teaser demo:
```
cd ~/Estimating-3D-Motion-Forces
source scripts/demo.sh
```

## Citation

If you find the code useful, please consider citing:
```bibtex
@InProceedings{li2019motionforcesfromvideo,
  author={Zongmian Li and Jiri Sedlar and Justin Carpentier and Ivan Laptev and Nicolas Mansard and Josef Sivic},
  title={Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

@article{li2022estimating,
  title={Estimating 3D motion and forces of human--Object interactions from internet videos},
  author={Li, Zongmian and Sedlar, Jiri and Carpentier, Justin and Laptev, Ivan and Mansard, Nicolas and Sivic, Josef},
  journal={International Journal of Computer Vision},
  volume={130},
  number={2},
  pages={363--383},
  year={2022},
  publisher={Springer}
}
```
