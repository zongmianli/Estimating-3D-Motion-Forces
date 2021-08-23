#include "bindings.h"
#include "../solver.h"

void ExposeSolver()
{
    bp::def("InitLogging", InitLogging);
    bp::def("Minimize", Minimize);
    bp::def("BuildLossFunction", BuildLossFunction);
    bp::def("SetOptions", SetOptions);
}
