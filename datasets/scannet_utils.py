# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Provides Python helper function to read My SUNRGBD dataset.

Author: Charles R. Qi
Date: October, 2017

Updated by Charles R. Qi
Date: December, 2018
Note: removed basis loading.
'''
import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points
import torch
# type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
# class2type = {type2class[t]:t for t in type2class}


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_camera_tensor(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = torch.clone(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2


class SUNObject3d(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.xmin = data[1] 
        self.ymin = data[2]
        self.xmax = data[1]+data[3]
        self.ymax = data[2]+data[4]
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        self.centroid = np.array([data[5],data[6],data[7]])
        self.unused_dimension = np.array([data[8],data[9],data[10]])
        self.w = data[8]
        self.l = data[9]
        self.h = data[10]
        self.orientation = np.zeros((3,))
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.heading_angle = -1 * np.arctan2(self.orientation[1], self.orientation[0])

def load_txt(path):
    R_path = path
    tmp = ''
    for line in open(R_path):
        tmp += line.rstrip() + ' '
    tmp = tmp.rstrip()
    lines = [tmp]
    Rtilt = np.array([float(x) for x in lines[0].split(' ')])
    Rtilt = np.reshape(Rtilt, (4, 4), order='F')
    Rtilt = Rtilt.transpose()
    return Rtilt

class SCANNET_Calibration(object):
    ''' Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''

    def __init__(self, calib_filepath, squence_name, if_tensor=True):
        R_path = calib_filepath + '/pose/' + squence_name + '.txt'#calib_filepath + '/pose/1.txt'  # change later, set 0.txt now.  contents:  camera_to_world matrix
        self.Rtilt = load_txt(R_path)
        if if_tensor:
            self.Rtilt_tensor = torch.from_numpy(self.Rtilt).to('cuda').to(torch.float32)

        K_path = calib_filepath + '/intrinsic/intrinsic_color.txt'
        self.K = load_txt(K_path)
        if if_tensor:
            self.K_tensor = torch.from_numpy(self.K).to('cuda').to(torch.float32)
        # K = np.array([float(x) for x in lines[1].split(' ')])
        # self.K = np.reshape(K, (3,3), order='F')
        # if if_tensor:
        #     self.K_tensor = torch.from_numpy(self.K).to('cuda').to(torch.float32)
        self.f_u = self.K[0,0]
        # self.f_u_tensor = torch.from_numpy(self.f_u).to('cuda')
        self.f_v = self.K[1,1]
        # self.f_v_tensor = torch.from_numpy(self.f_v).to('cuda')
        self.c_u = self.K[0,2]
        # self.c_u_tensor = torch.from_numpy(self.c_u).to('cuda')
        self.c_v = self.K[1,2]
        # self.c_v_tensor = torch.from_numpy(self.c_v).to('cuda')
   
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        new_pc = np.ones((pc.shape[0], 4))
        new_pc[:, :3] = pc
        extra_matrix = np.linalg.inv(self.Rtilt)
        # extra_matrix = self.Rtilt
        pc2 = np.dot(extra_matrix, np.transpose(new_pc)) # (3,n)
        pc2 = np.transpose(pc2[:-1, :])
        return pc2 #flip_axis_to_camera(pc2)

    def project_upright_depth_to_camera_tensor(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        new_pc = torch.ones((pc.shape[0], 4), device=pc.device)
        new_pc[:, :3] = pc
        # pc2 = torch.mm(torch.transpose(self.Rtilt_tensor, 0, 1), torch.transpose(pc[:,0:3], 0, 1)) # (3,n)
        extra_matrix = torch.linalg.inv(self.Rtilt_tensor)
        # pc2 = torch.mm(torch.transpose(extra_matrix, 0, 1), torch.transpose(new_pc, 0, 1))  # (3,n)
        pc2 = torch.mm(extra_matrix, torch.transpose(new_pc, 0, 1))  # (3,n)
        pc2 = torch.transpose(pc2[:-1, :], 0, 1)
        # return flip_axis_to_camera_tensor(torch.transpose(pc2[:,:]/pc2[-1,:], 0, 1))
        return pc2 #flip_axis_to_camera_tensor(pc2)

    def project_upright_depth_to_image(self, pc, trans_mtx=None):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        # print(self.K.shape)
        intrinsics_matrix = self.K[:-1, :-1]
        # intrinsics_matrix = np.linalg.inv(intrinsics_matrix)
        uv = np.dot(intrinsics_matrix, np.transpose(pc2)) # (n,3)
        uv = np.transpose(uv)
        d = uv[:, -1].copy()
        uv[:, :-1] = uv[:, :-1] / (np.expand_dims(uv[:, -1], -1) + 1e-32)
        # print('====================')
        # print(uv[:,0:2])
        if trans_mtx is not None:
            # print(uv[:,0:2].shape)
            # print(trans_mtx.shape)
            # print(uv[:,0:2])
            # print('INNNNNNNNNNNNNNNNN')
            uv[:,0:2] = np.dot(uv[:,0:2], trans_mtx)
        # print('In colib')
        # print(uv)
        # print('--------------------')

        # uv[:,0] /= (uv[:,2]+1e-32)
        # uv[:,1] /= (uv[:,2]+1e-32)
        # print(uv[:, 0:2])

        return uv[:,0:2], pc2[:,2], d

    def project_upright_depth_to_image_tensor(self, pc, trans_mtx=None):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera_tensor(pc)
        # print(self.K.shape)
        intrinsics_matrix = self.K_tensor[:-1, :-1]
        # uv = torch.mm(pc2, torch.transpose(self.K_tensor, 0, 1)) # (n,3)
        uv = torch.mm(intrinsics_matrix, torch.transpose(pc2, 0, 1)) # (n,3)
        uv = torch.transpose(uv, 1, 0)
        d = uv[:, -1].clone()
        uv[:, :-1] = uv[:, :-1] / (torch.unsqueeze(uv[:, -1], -1)+1e-32)
        # uv = uv[:, :-1]
        # print('====================')
        # print(uv[:,0:2])
        if trans_mtx is not None:
            # print(uv[:,0:2].shape)
            # print(trans_mtx.shape)
            # print(uv[:,0:2])
            # print('INNNNNNNNNNNNNNNNN')
            uv[:,0:2] = torch.mm(uv[:,0:2], trans_mtx)
        # print('In colib')
        # print(uv)
        # print('--------------------')

        # uv[:,0] /= (uv[:,2]+1e-32)
        # uv[:,1] /= (uv[:,2]+1e-32)
        # print(uv[:, 0:2])

        return uv[:,0:2], pc2[:,2], d

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)

 
 
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def rotz_tensor(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.Tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """ 
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_sunrgbd_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [SUNObject3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_depth_points(depth_filename):
    depth = np.loadtxt(depth_filename)
    return depth

def load_depth_points_mat(depth_filename):
    depth = sio.loadmat(depth_filename)['instance']
    return depth

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
 
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)


def compute_box_3d_resize(box_center, box_angle, box_size, calib, trans_mtx):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = box_center
    size = box_size / 2
    # box_angle = box_angle % (np.pi * 2)
    # make center upright
    # center[..., [0,1,2]] = center[..., [0,2,1]]
    # center[..., 1] *= -1

    # compute rotational matrix around yaw axis
    R = rotz(-1 * box_angle)
    # b,a,c = dimension
    # print R, a,b,c

    # 3d bounding box dimensions
    l = size[0]  # along heading arrow
    w = size[1]  # perpendicular to heading arrow
    h = size[2]

    # rotate and translate 3d bounding box
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d, _ = calib.project_upright_depth_to_image(np.transpose(corners_3d), trans_mtx)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_box_3d_offset_tensor(box_center, box_angle, box_size, calib, x_offset, y_offset):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = box_center
    size = box_size / 2
    # box_angle = box_angle % (np.pi * 2)
    # make center upright
    # center[..., [0,1,2]] = center[..., [0,2,1]]
    # center[..., 1] *= -1

    # compute rotational matrix around yaw axis
    # R = rotz_tensor(-1 * box_angle).to('cuda')
    R = rotz_tensor(box_angle).to('cuda')
    # b,a,c = dimension
    # print R, a,b,c

    # 3d bounding box dimensions
    l = size[0]  # along heading arrow
    w = size[1]  # perpendicular to heading arrow
    h = size[2]

    # rotate and translate 3d bounding box
    x_corners = torch.Tensor([-l, l, l, -l, -l, l, l, -l]).to('cuda')
    y_corners = torch.Tensor([w, w, -w, -w, w, w, -w, -w]).to('cuda')
    z_corners = torch.Tensor([h, h, h, h, -h, -h, -h, -h]).to('cuda')
    corners_stack = torch.vstack((x_corners, y_corners, z_corners))
    corners_3d = torch.mm(R, corners_stack).to('cuda')
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d, _, d = calib.project_upright_depth_to_image_tensor(torch.transpose(corners_3d, 0, 1))
    # print('========================')
    # print(x_offset)
    # print(y_offset)
    # print(corners_2d)
    corners_2d[:,0] += y_offset
    corners_2d[:,1] += x_offset

    # print 'corners_2d: ', corners_2d
    return corners_2d, torch.transpose(corners_3d, 0, 1), d

def compute_box_3d_offset(box_center, box_angle, box_size, calib, x_offset, y_offset):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = box_center
    size = box_size / 2
    # box_angle = box_angle % (np.pi * 2)
    # make center upright
    # center[..., [0,1,2]] = center[..., [0,2,1]]
    # center[..., 1] *= -1

    # compute rotational matrix around yaw axis
    # R = rotz(-1 * box_angle)
    R = rotz(box_angle)
    # b,a,c = dimension
    # print R, a,b,c

    # 3d bounding box dimensions
    l = size[0]  # along heading arrow
    w = size[1]  # perpendicular to heading arrow
    h = size[2]

    # rotate and translate 3d bounding box
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d, _, d = calib.project_upright_depth_to_image(np.transpose(corners_3d))
    # print('========================')
    # print(x_offset)
    # print(y_offset)
    # print(corners_2d)
    corners_2d[:,0] += y_offset
    corners_2d[:,1] += x_offset

    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d), d

def compute_box_3d(box_center, box_angle, box_size, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = box_center 
    size = box_size / 2
    #box_angle = box_angle % (np.pi * 2)
    # make center upright
    #center[..., [0,1,2]] = center[..., [0,2,1]]
    #center[..., 1] *= -1

    # compute rotational matrix around yaw axis
    R = rotz(-1*box_angle)
    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = size[0] # along heading arrow
    w = size[1] # perpendicular to heading arrow
    h = size[2]

    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d,_ = calib.project_upright_depth_to_image(np.transpose(corners_3d))
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_box_3d_bp(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = rotz(-1*obj.heading_angle)
    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = obj.l # along heading arrow
    w = obj.w # perpendicular to heading arrow
    h = obj.h

    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d,_ = calib.project_upright_depth_to_image(np.transpose(corners_3d))
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in image coord.
            orientation_3d: (2,3) array in depth coord.
    '''
    
    # orientation in object coordinate system
    ori = obj.orientation
    orientation_3d = np.array([[0, ori[0]],[0, ori[1]],[0,0]])
    center = obj.centroid
    orientation_3d[0,:] = orientation_3d[0,:] + center[0]
    orientation_3d[1,:] = orientation_3d[1,:] + center[1]
    orientation_3d[2,:] = orientation_3d[2,:] + center[2]
    
    # project orientation into the image plane
    orientation_2d,_ = calib.project_upright_depth_to_image(np.transpose(orientation_3d))
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image


import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def flip_axis_to_camera_batch_tensor(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = torch.clone(pc)
    pc2[:, :, [0,1,2]] = pc2[:, :, [0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:, :, 1] *= -1
    return pc2

def project_3dpoint_to_2dpoint_tensor(pc, K_tensor=None, Rtilt_tensor=None, trans_mtx=None):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    pc2 = torch.bmm(torch.transpose(Rtilt_tensor, 1, 2), torch.transpose(pc[:, :, 0:3], 1, 2))
    pc2 = flip_axis_to_camera_batch_tensor(torch.transpose(pc2, 1, 2))
    # print(self.K.shape)
    uv = torch.bmm(pc2, torch.transpose(K_tensor, 1, 2)) # (n,3)
    # print('====================')
    # print(uv[:,0:2])
    if trans_mtx is not None:
        # print(uv[:,0:2].shape)
        # print(trans_mtx.shape)
        # print(uv[:,0:2])
        # print('INNNNNNNNNNNNNNNNN')
        uv[:,:,0:2] = torch.bmm(uv[:,:,0:2], trans_mtx)
        # print('In colib')
        # print(uv)
        # print('--------------------')

    uv[:, :, 0] /= (uv[:, :, 2]+1e-32)
    uv[:, :, 1] /= (uv[:, :, 2]+1e-32)
        # print(uv[:, 0:2])

    return uv[:, :, 0:2]

def flip_axis_to_camera_batch_corners_tensor(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = torch.clone(pc)
    pc2[:, :, :, [0,1,2]] = pc2[:, :, :, [0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:, :, :, 1] *= -1
    return pc2

def project_3dpoint_to_2dpoint_corners_tensor(pc, K_tensor=None, Rtilt_tensor=None, trans_mtx=None):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    K_tensor = K_tensor.unsqueeze(1)
    extra_matrix = torch.linalg.inv(Rtilt_tensor)
    extra_matrix = extra_matrix.unsqueeze(1)

    new_pc = torch.ones((pc.shape[0], pc.shape[1], pc.shape[2], 4), dtype=pc.dtype ,device=pc.device)
    new_pc[:, :, :, :3] = pc
    # extra_matrix = np.linalg.inv(self.Rtilt)
    # extra_matrix = self.Rtilt
    # pc2 = np.dot(extra_matrix, np.transpose(new_pc))  # (3,n)
    # pc2 = np.transpose(pc2[:-1, :])
    pc2 = torch.matmul(extra_matrix,  torch.transpose(new_pc[:, :, :, 0:4], 2, 3))
    pc2 = torch.transpose(pc2, 2, 3)[:, :, :, :-1]
    # pc2 = flip_axis_to_camera_batch_corners_tensor(torch.transpose(pc2, 2, 3))
    # print(self.K.shape)
    intrinsics_matrix = K_tensor[:, :, :-1, :-1]
    # intrinsics_matrix = np.linalg.inv(intrinsics_matrix)
    # uv = np.dot(intrinsics_matrix, np.transpose(pc2))  # (n,3)
    # uv = torch.matmul(pc2, torch.transpose(K_tensor, 2, 3)) # (n,3)
    uv = torch.matmul(intrinsics_matrix, torch.transpose(pc2, 2, 3))  # (n,3)
    uv = torch.transpose(uv, 2, 3)
    # print('====================')
    # print(uv[:,0:2])
    if trans_mtx is not None:
        # print(uv[:,0:2].shape)
        # print(trans_mtx.shape)
        # print(uv[:,0:2])
        # print('INNNNNNNNNNNNNNNNN')
        uv[:,:,0:2] = torch.matmul(uv[:,:,0:2], trans_mtx)
        # print('In colib')
        # print(uv)
        # print('--------------------')

    uv[:, :, :, 0] /= (uv[:, :, :, 2]+1e-32)
    uv[:, :, :, 1] /= (uv[:, :, :, 2]+1e-32)
    d = uv[:, :, :, 2]
        # print(uv[:, 0:2])

    return uv[:, :, :, 0:2], d

def corners_project_upright_depth_to_camera_tensor(self, pc, Rtilt_tensor):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = torch.mm(torch.transpose(Rtilt_tensor, 0, 1), torch.transpose(pc[:,0:3], 0, 1)) # (3,n)

        return flip_axis_to_camera_batch_corners_tensor(torch.transpose(pc2, 0, 1))

def corners_project_upright_depth_to_image_tensor(self, pc, K_tensor=None, Rtilt_tensor=None, trans_mtx=None):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    pc2 = corners_project_upright_depth_to_camera_tensor(pc, Rtilt_tensor)
    # print(self.K.shape)
    uv = torch.mm(pc2, torch.transpose(K_tensor, 0, 1)) # (n,3)
    # print('====================')
    # print(uv[:,0:2])
    if trans_mtx is not None:
        # print(uv[:,0:2].shape)
        # print(trans_mtx.shape)
        # print(uv[:,0:2])
        # print('INNNNNNNNNNNNNNNNN')
        uv[:,0:2] = torch.mm(uv[:,0:2], trans_mtx)
        # print('In colib')
        # print(uv)
        # print('--------------------')

    uv[:,0] /= (uv[:,2]+1e-32)
    uv[:,1] /= (uv[:,2]+1e-32)
    # print(uv[:, 0:2])

    return uv[:,0:2], pc2[:,2]


# def project_3dpoint_to_2dpoint_corners_tensor(box_center, box_angle, box_size, K_tensor=None, Rtilt_tensor=None):
#     ''' Takes an object and a projection matrix (P) and projects the 3d
#         bounding box into the image plane.
#         Returns:
#             corners_2d: (8,2) array in image coord.
#             corners_3d: (8,3) array in in upright depth coord.
#     '''
#     center = box_center
#     size = box_size / 2
#     # box_angle = box_angle % (np.pi * 2)
#     # make center upright
#     # center[..., [0,1,2]] = center[..., [0,2,1]]
#     # center[..., 1] *= -1
#
#     # compute rotational matrix around yaw axis
#     R = rotz_tensor(-1 * box_angle).to('cuda')
#     # b,a,c = dimension
#     # print R, a,b,c
#
#     # 3d bounding box dimensions
#     l = size[0]  # along heading arrow
#     w = size[1]  # perpendicular to heading arrow
#     h = size[2]
#
#     # rotate and translate 3d bounding box
#     x_corners = torch.Tensor([-l, l, l, -l, -l, l, l, -l]).to('cuda')
#     y_corners = torch.Tensor([w, w, -w, -w, w, w, -w, -w]).to('cuda')
#     z_corners = torch.Tensor([h, h, h, h, -h, -h, -h, -h]).to('cuda')
#     corners_stack = torch.vstack((x_corners, y_corners, z_corners))
#     corners_3d = torch.mm(R, corners_stack).to('cuda')
#     corners_3d[0, :] += center[0]
#     corners_3d[1, :] += center[1]
#     corners_3d[2, :] += center[2]
#
#     # project the 3d bounding box into the image plane
#     corners_2d, _ = corners_project_upright_depth_to_image_tensor(torch.transpose(corners_3d, 0, 1), K_tensor=K_tensor, Rtilt_tensor=Rtilt_tensor)
#     # print('========================')
#     # print(x_offset)
#     # print(y_offset)
#     # print(corners_2d)
#     # corners_2d[:,0] += y_offset
#     # corners_2d[:,1] += x_offset
#
#     # print 'corners_2d: ', corners_2d
#     return corners_2d, torch.transpose(corners_3d, 0, 1)