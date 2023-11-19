import numpy as np
import numpy.linalg as LA
import pickle as pk
import chumpy as ch
import xml.etree.ElementTree as ET
from glob import glob
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp,log


def rgb_to_hex(color, prefix="0x"):
    '''
    Converting a RGB color tuple to a six digit code
    '''
    color = [int(c) for c in color]
    r = color[0]
    g = color[1]
    b = color[2]
    return prefix+"{0:02x}{1:02x}{2:02x}".format(r, g, b)


def is_float(s):
    '''
    Tests if the input string, s, can be converted to a float number.
    '''
    try:
        x = float(s)
    except ValueError:
        return False
    else:
        return True


def is_int(s):
    '''
    Tests if the input string, s, can be converted to a integer without rounding.
    '''
    try:
        x = float(s)
        n = int(x)
    except ValueError:
        return False
    else:
        return n == x


def load_parameters_from_txt(file_path):
    '''
    This helper function reads the lines of the given input txt file,
    and converts the content of the input file to a Python dict.
    The input txt file must be formatted as required:
    a. Each line starts with the name of a parameter, and is followed by
       a string of values split by comma. There should be a space between the
       parameter name and the values.
    b. Values in the string that can be converted to numbers will be converted
       to integers by defaults, unless they contain a point character '.' or
       the exp character 'e'. They will be converted to floats in this case.
       Otherwise the values will be kept as strings in the output dict.
    c. No space or tab is allowed at the end of each line.
    --
    Example:

    In example.txt:
    float_list 0.7,0.1,2e1
    int_list 200,12
    str_list thisisstring,0..1
    mixted_list 542,2e-1,.1,adf1,12w

    In [8]: params = load_parameters_from_txt('example.txt')
    In [9]: params
    Out[9]:
    {'float_list': [0.7, 0.1, 20.0],
     'int_list': [200, 12],
     'mixted_list': [542, 0.2, 0.1, 'adf1', '12w'],
     'str_list': ['thisisstring', '0..1']}
    '''
    with open(file_path, 'r') as f:
        params = dict()
        for line in f.read().splitlines():
            name, values_string = line.split(' ')
            values = []
            for s in values_string.split(','):
                if is_float(s):
                    if '.' in s or 'e' in s:
                        values.append(float(s))
                    else:
                        values.append(int(s))
                else:
                    values.append(s)
            params[name] = values
        return params


def findMinAngle(angle, option=2):
	'''
	returns an angle value between -pi and +pi that achieve the same rotation
	than the input angle.
	'''
	angle = angle % (2*np.pi)
	if option==2:
		if angle > np.pi:
			angle -= 2*np.pi
	return angle


def normalizeRotationAngles(seqConfig, decoration, align_opt=1, angle_opt=1):
	'''
	find minimal rotation angle values (between -pi and +pi) for the
	axis-angle blocks and revolute joints in seqConfig.
	'''
	normalizedConfig = seqConfig.copy().getA()
	njoints = decoration.shape[0]
	nf = seqConfig.shape[1]
	for j in range(njoints):
		j_type = decoration[j,2]
		j_idx = decoration[j,3]
		if j_type==1: # free-floating joint
			joints = range(j_idx+3, j_idx+6)
			norms = LA.norm(seqConfig[joints,:], axis=0)
			prev_axis = np.array([0.,0.,-1.])
			for i in range(nf):
				if norms[i]>=1e-30:
					# skip to next frame if current frame is empty
					axisMat = seqConfig[joints, i]/norms[i]
					axis = axisMat.getA().reshape(-1)
					angle = findMinAngle(norms[i], option=angle_opt)
					if align_opt==1:
						if axis[0]<0: # let axis point to x+ plan
							axis = -1.*axis
							if angle_opt==1:
								angle = 2.*np.pi - angle
							elif angle_opt==2:
								angle = angle*(-1.)
					elif align_opt==2:
						# align current axis with previous one
						if np.matrix(prev_axis)*np.matrix(axis).T < 0.:
							axis = -1.*axis
							if angle_opt==1:
								angle = 2.*np.pi - angle
							elif angle_opt==2:
								angle = angle*(-1.)
						prev_axis = axis.copy()
					normalizedConfig[joints, i] = axis*angle
		elif j_type==2: # spherical joint
			joints = range(j_idx, j_idx+3)
			norms = LA.norm(seqConfig[joints,:], axis=0)
			for i in range(nf):
				axisMat = seqConfig[joints, i]/norms[i]
				axis = axisMat.getA().reshape(-1)
				angle = findMinAngle(norms[i], option=2)
				if axis[0]<0: # let axis point to x+ plan
					axis = -1*axis
					angle = angle*(-1.)
				normalizedConfig[joints, i] = axis*angle
		elif j_type in [3,4,5]: # revolute joints RX, RY, or RZ
			for i in range(nf):
				normalizedConfig[j_idx, i] = \
					findMinAngle(seqConfig[j_idx, i], option=2)
	return np.matrix(normalizedConfig)


def rotation_matrix(a, b):
	'''compute a rotation matrix from vector a to vector b'''
	a = np.matrix(a)
	b = np.matrix(b)
	if a.shape != (3,1):
		a = a.T
	if b.shape != (3,1):
		b = b.T
	v = cross(a, b)
	c = a.T*b # cos of angle, note that the result is a 1x1 matrix
	v_cross = cross_3d(v)
	R = eye(3) + v_cross + v_cross*v_cross*(1/(1+c[0,0]))
	return R


def groundPointNormalToPinoConfig(p, normal):
	vec0 = np.array([0.,-1.,0.]) # original ground normal
	vec1 = normal/LA.norm(normal)
	rotationMat = rotation_matrix(vec0, vec1)
	M = se3.SE3(rotationMat, np.matrix(p).T)
	return np.matrix(se3ToXYZQUAT(M)).T


def cross_3d(v):
	'''given 3d vector v this function outputs v_cross matrix'''
	v = np.matrix(v)
	if v.shape != (3,1):
		v = v.T
	v_cross = np.matrix([[0., -v[2,0], v[1,0]],
					 	 [v[2,0], 0., -v[0,0]],
						 [-v[1,0], v[0,0], 0.]])
	return v_cross


def axisAngleToRotationMat(axis, angle):
	sina = np.sin(angle)
	cosa = np.cos(angle)
	axis = axis/LA.norm(axis)
	# rotation matrix around unit vector
	R = np.diag([cosa, cosa, cosa])
	R += np.outer(axis, axis) * (1.0 - cosa)
	axis *= sina
	R += np.array( [[ 0.0,    -axis[2], axis[1]],
					[ axis[2], 0.0,    -axis[0]],
					[-axis[1], axis[0], 0.0]])
	return R


def getPaths(video, frameAB, repo_dir=None, mode=2):
	video = video.split('.')[0] # remove extension
	clipName = video+'_f'+str(frameAB[0])+'-'+str(frameAB[1])
	dataDir = join(repo_dir, 'smplify_m{}'.format(mode))
	posePath = join(dataDir, clipName+'_smpl.pkl')
	shapePath = join(dataDir, clipName+'_smpl_shape.npz')
	xmlDir = join(repo_dir, 'annotation', clipName, 'info')
	xmlPaths = sorted(glob(join(xmlDir, '*.xml')))
	outPath = join(repo_dir, 'results' , clipName+'_smpl_pino.pkl')
	return posePath, shapePath, xmlPaths, outPath


def getJointEvalIdx(toolCttIdx={-1}, gdCttIdx={-1}):
	jointEvalIdx = {'leftLeg': {4,7,10},
					'rightLeg': {5,8,11},
					'spine': {0,1,2,3,6,9,12,13,14,15},
					'leftArm': {16,18,20,22},
					'rightArm': {17,19,21,23},
					'all': set(range(24))}
	jointEvalIdx['legs'] = jointEvalIdx['leftLeg']|jointEvalIdx['rightLeg']
	jointEvalIdx['arms'] = jointEvalIdx['leftArm']|jointEvalIdx['rightArm']
	jointEvalIdx['toolContact'] = toolCttIdx
	jointEvalIdx['gdContact'] = gdCttIdx
	jointEvalIdx['contact'] = jointEvalIdx['toolContact']|jointEvalIdx['gdContact'] - {-1}
	jointEvalIdx['nonContact'] = jointEvalIdx['all'] - jointEvalIdx['contact']
	return jointEvalIdx


def getAnnotationFromXML(XMLPath, objPts=None, groundPts=None, bodyJoints=None):
	if bodyJoints is None:
		bodyJoints = {'R_Ankle':0, 'R_Knee':1, 'R_Hip':2,
					  'L_Hip':3, 'L_Knee':4, 'L_Ankle':5,
					  'R_Wrist':6, 'R_Elbow':7, 'R_Shoulder':8,
					  'L_Shoulder':9, 'L_Elbow':10, 'L_Wrist':11}
	objPts = {'obj_root':0} if objPts is None else objPts
	groundPts = {'gd_0':0,'gd_1':1,'gd_2':2} if groundPts is None else groundPts
	annotation = ET.parse(XMLPath).getroot()
	keypoints = annotation.find('keypoints')
	bodyPos = np.zeros((12,3))
	objPos = np.zeros((len(objPts.keys()),3))
	groundPos = np.zeros((3,3))
	hasObjInfo = False
	hasGroundInfo = False
	for keypoint in keypoints.findall('keypoint'):
		name = keypoint.get('name')
		x = float(keypoint.get('x'))
		y = float(keypoint.get('y'))
		z = -1.*float(keypoint.get('z')) # note, that the z should be swapped
		if name in bodyJoints.keys():
			bodyPos[bodyJoints[name]] = np.array([x,y,z])
		if name in objPts.keys():
			objPos[objPts[name]] = np.array([x,y,z])
			hasObjInfo = True
		if name in groundPts.keys():
			groundPos[groundPts[name]] = np.array([x,y,z])
			hasGroundInfo = True

	outDict = {'bodyPos': bodyPos}
	if hasGroundInfo:
		#gdNormal = np.cross(groundPos[1]-groundPos[0], groundPos[2]-groundPos[0])
		#gdPt = groundPos[0]
		#outDict['gdNormal'] = gdNormal
		#outDict['gdPt'] = gdPt
		outDict['groundPos'] = groundPos
	if hasObjInfo:
		outDict['objPos'] = objPos
	return outDict


def procrustes(A, B):
    transposed = False
    if A.shape[0]!=3:
        A = A.T
        B = B.T
        transposed = True
    N = A.shape[1]
    assert(B.shape==(3,N))
    a_bar = A.mean(axis=1, keepdims=True)
    b_bar = B.mean(axis=1, keepdims=True)
    A_c = A - a_bar
    B_c = B - b_bar
    M = A_c.dot(B_c.T)
    U, Sigma, Vh = LA.svd(M)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1,-1] = LA.det(V)*LA.det(U)
    R = V.dot(Z.dot(U.T))
    s = np.trace(R.dot(M)) / np.trace(A_c.T.dot(A_c))
    t = b_bar - s*(R.dot(a_bar))
    A_hat = s*(R.dot(A)) + t
    if transposed:
        A_hat = A_hat.T
    return (R, t, s, A_hat)


def load_3d_human_pose_from_smpl(video_name, gender, shape_path, j3d_path):
	'''
	Get the subject's joint 3D positions from SMPLify-CoM outputs.
	'''
	from smpl_webuser.serialization import load_model
	# load SMPL model accroding to gender
	if gender == 'male':
		model_path = join(
			'person_models', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
	elif gender == 'female':
		model_path = join(
			'person_models', 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
	else:
		raise ValueError('Unknown gender: {}'.format(gender))
	smpl_model = load_model(model_path)

	j3dCanoPos = ch.zeros((24, 3))
	with open(shape_path, 'r') as fshape:
		data = pk.load(fshape)
		betas = data[video_name][0] # use the shape vec at first frame
		n_betas = betas.shape[0]
		# get canonical 3D position of each joint
		j3dDirs = np.dstack([smpl_model.J_regressor.dot(
			smpl_model.shapedirs[:,:,i]) for i in range(n_betas)])
		j3dCanoPos = ch.array(j3dDirs).dot(betas) + \
			smpl_model.J_regressor.dot(smpl_model.v_template.r)
		print('load j3dCanoPos for *{0}*'.format(video_name))
	j3d_pos = j3dCanoPos.r
	# save j3d_pos to file
	if exists(j3d_path):
		with open(j3d_path, 'r') as fj3d:
			data = pk.load(fj3d)
			data[video_name] = j3d_pos
	else:
		data = {video_name: j3d_pos}
		if not exists(dirname(j3d_path)):
			makedirs(dirname(j3d_path))
	with open(j3d_path, 'w') as outfj3d:
		pk.dump(data, outfj3d)
		print('save j3dCanoPos to {0}'.format(j3d_path))
	return j3d_pos


def load_3d_human_pose_from_file(video_name, j3d_path):
	with open(j3d_path, 'r') as fj3d:
		data = pk.load(fj3d)
		j3d_pos = data[video_name]
		return j3d_pos


def load_config_smplify_com(video_name,
							pose_path,
							cam_path,
							frame_start,
							frame_end):
	'''
	load human initial pose trajectory from SMPLify-CoM outputs.
	'''
	seqPoses = None
	seqTrans = None
	seqCam = None
	with open(pose_path, 'r') as fpose:
		data = pk.load(fpose)
		seqPoses = data[video_name][frame_start:frame_end]
		print('pose data loaded from {}'.format(pose_path))
	depth_offset = 0.
	with open(cam_path, 'r') as fcam:
		data = pk.load(fcam)
		seqCam = data[video_name]
		seqTrans = seqCam['cam_t'][frame_start:frame_end].copy()
		# center the depth to zero
		depth_offset = np.mean(seqTrans[:,2])
		seqTrans[:,2] -= depth_offset
		print('camera loaded from {}'.format(cam_path))
	seqConfigSMPL = np.concatenate((seqTrans, seqPoses), axis=1)
	seqConfigSMPL = np.matrix(seqConfigSMPL).T
	# convert seqConfigSMPL to seqConfig
	idxPino = [ 0, 1, 5, 9, 2, 6,
			   10, 3, 7,11, 4, 8,
			   12,14,19,13,15,20,
			   16,21,17,22,18,23]
	seqConfig = np.matrix(np.zeros(seqConfigSMPL.shape))
	# copy the first 3 rows of configSMPL to config
	seqConfig[:3,:] = seqConfigSMPL[:3,:]
	# relocate the rest of the rows (axis angles)
	for jSMPL in range(24):
		j = idxPino[jSMPL]
		seqConfig[(3*j+3):(3*(j+1)+3), :] = seqConfigSMPL[(3*jSMPL+3):(3*(jSMPL+1)+3), :]
	# get camera parameters (assuming static camera)
	cam = {'cam_t': np.zeros(3),
		   'cam_f': seqCam['cam_f'][frame_start],
		   'cam_rt': seqCam['cam_rt'][frame_start],
		   'cam_c': seqCam['cam_c'][frame_start],
		   'cam_k': seqCam['cam_k'][frame_start],
		   'depth_offset': depth_offset}
	return seqConfig, cam


def set_meshcat_camera_view(gui, camera_angle=0):
    '''
    Adjust the camera view of the Meshcat GUI
    '''
    R1 = se3.AngleAxis(np.pi/2., np.matrix([0.,0.,-1.]).T)
    R2 = se3.AngleAxis(np.pi/2., np.matrix([1.,0.,0.]).T)
    #R3 = se3.AngleAxis(camera_angle, np.matrix([0.,1.,0.]).T).matrix()
    R = R2*R1
    t = np.matrix([0., 0., 0.]).T
    oMcam = se3.SE3(R.matrix(), t)
    gui["/Cameras/default"].set_transform(
        np.array(oMcam.homogeneous))
