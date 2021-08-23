# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_solver_HEADERS
    camera.h
    pose_prior_gmm.h
    solver.h
    dataloader/dataloader.h
    dataloader/dataloader_person.h
    dataloader/dataloader_object.h
    cost_functors/contact_ground_spatial.h
    cost_functors/contact_ground_temporal.h
    cost_functors/contact_object_spatial.h
    cost_functors/contact_object_temporal.h
    cost_functors/contact_symmetric.h
    cost_functors/contact_temporal.h
    cost_functors/horizontal_object.h
    cost_functors/limit_ground_force.h
    cost_functors/limit_object_force.h
    cost_functors/object_3d_errors.h
    cost_functors/object_acceleration.h
    cost_functors/object_data.h
    cost_functors/object_torque.h
    cost_functors/object_velocity.h
    cost_functors/person_3d_errors.h
    cost_functors/person_acceleration.h
    cost_functors/person_center_of_mass.h
    cost_functors/person_data.h
    cost_functors/person_depth.h
    cost_functors/person_pose.h
    cost_functors/person_torque.h
    cost_functors/person_velocity.h
    cost_functors/smoothing_cartesian_velocity.h
    cost_functors/smoothing_ground_force.h
    cost_functors/smoothing_object_force.h
    bindings/bindings.h
)

SET(${PROJECT_NAME}_solver_SOURCES
    build_loss_function.cc
    camera.cc
    pose_prior_gmm.cc
    set_options.cc
    solver.cc
    dataloader/dataloader.cc
    dataloader/dataloader_person.cc
    dataloader/dataloader_object.cc
    cost_functors/object_3d_errors.cc
    cost_functors/object_torque.cc
    cost_functors/person_3d_errors.cc
    cost_functors/person_data.cc
    cost_functors/person_torque.cc
    cost_functors/contact_ground_spatial.cc
    cost_functors/contact_temporal.cc
    cost_functors/limit_ground_force.cc
    cost_functors/limit_object_force.cc
    bindings/bindings.cc
    bindings/expose_camera.cc
    bindings/expose_dataloader.cc
    bindings/expose_dataloader_person.cc
    bindings/expose_dataloader_object.cc
    bindings/expose_pose_prior_gmm.cc
    bindings/expose_solver.cc
)

ADD_LIBRARY(solver SHARED ${${PROJECT_NAME}_solver_SOURCES} ${${PROJECT_NAME}_solver_HEADERS})
PKG_CONFIG_USE_DEPENDENCY(solver eigenpy)
PKG_CONFIG_USE_DEPENDENCY(solver pinocchio)

TARGET_LINK_LIBRARIES(solver ceres)
TARGET_LINK_BOOST_PYTHON(solver)

SET_TARGET_PROPERTIES(solver PROPERTIES PREFIX "../../lib/")

# Change the extension for python bindings on MacOS
IF(APPLE)
  SET_TARGET_PROPERTIES(solver PROPERTIES SUFFIX ".so")
ENDIF(APPLE)
