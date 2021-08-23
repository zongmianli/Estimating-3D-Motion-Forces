#ifndef __MODULE_H__
#define __MODULE_H__

#include <Eigen/Core>
#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/geometry.hpp>

#define BOOST_PYTHON_MAX_ARITY 30

namespace bp = boost::python;

void ExposeCamera();
void ExposeDataloader();
void ExposeDataloaderPerson();
void ExposeDataloaderObject();
void ExposePosePriorGmm();
void ExposeSolver();

#endif // #ifndef __MODULE_H__
