#include "bindings.h"

// Wrap the C++ functions and classes in a Python module named "solver"
BOOST_PYTHON_MODULE(solver)
{
    eigenpy::enableEigenPy();

    ExposeCamera();
    ExposeDataloader();
    ExposeDataloaderPerson();
    ExposeDataloaderObject();
    ExposePosePriorGmm();
    ExposeSolver();
}
