import cv2
import numpy as np
import chumpy as ch

from os.path import join, exists, abspath, dirname, basename

from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated

MODEL_DIR = '../models'
MODEL_FEMALE_PATH = join(
    MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
MODEL_MALE_PATH = join(MODEL_DIR,
                       'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
model_paths = [MODEL_FEMALE_PATH, MODEL_MALE_PATH]

betas = np.zeros(10)
modelnames = ['female','male']
jointPos = {}
for idx in range(2):
    model = load_model(model_paths[idx])
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i]) for i in range(len(betas))])
    jointPos[modelnames[idx]] = np.array(Jdirs).dot(betas) + model.J_regressor.dot(model.v_template.r)
    print type(jointPos[modelnames[idx]])

print jointPos.keys()
print jointPos['female'].shape
print jointPos['male'].shape
print type(jointPos)
print dir(jointPos)
np.savez_compressed('j3d_positions.npz', jointPos=jointPos)
