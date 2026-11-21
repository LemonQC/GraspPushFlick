"""Explicit upstream GraspNet 17-column pose -> source SR5 TCP mapping.

This is an interface, not a replacement for a trained GraspNet detector. Inputs
must be actual external predictions and the matching optical calibration.
"""
import numpy as np
from scipy.spatial.transform import Rotation

GRASP_FROM_TCP = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])


def prediction_to_tcp(row, T_world_optical, tcp_offset_m=0.):
    row, t = np.asarray(row,float), np.asarray(T_world_optical,float)
    if row.shape!=(17,) or t.shape!=(4,4) or not np.isfinite(np.r_[row,t.ravel(),tcp_offset_m]).all():
        raise ValueError('Expected finite upstream 17-column row and 4x4 optical transform')
    if not 0<row[1]<=.08:
        raise ValueError('Predicted total opening exceeds PGIA 80mm range')
    if not np.allclose(t[3],[0,0,0,1],atol=1e-7):
        raise ValueError('Invalid homogeneous optical transform')
    r=row[4:13].reshape(3,3)
    for rotation in (r,t[:3,:3]):
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rotation),1,atol=1e-4):
            raise ValueError('Rotation must be proper SO(3); no silent projection/correction')
    rwg=t[:3,:3]@r
    p=t[:3,:3]@row[13:16]+t[:3,3]+tcp_offset_m*rwg[:,0]
    q=Rotation.from_matrix(rwg@GRASP_FROM_TCP).as_quat()
    return {'xyz_m':p.tolist(),'quat_xyzw':q.tolist(),'opening_m':float(row[1]),'score':float(row[0]),
            'tcp_frame':'ral_tcp','tcp_offset_m':float(tcp_offset_m),
            'calibration_status':'SOURCE_APPROX_NOT_HARDWARE_CALIBRATED',
            'mapping':'TCP +Z=GraspNet +X approach; TCP +X=GraspNet +Y closing axis'}
