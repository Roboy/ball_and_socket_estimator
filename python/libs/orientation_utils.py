import torch
import numpy as np

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cpu()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
        
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0
        
    out_euler=torch.autograd.Variable(torch.zeros(batch,3).cpu())
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler

#euler batch*4
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]

    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1
    c2=torch.cos(euler[:,1]).view(batch,1)#batch*1
    s2=torch.sin(euler[:,1]).view(batch,1)#batch*1
    c3=torch.cos(euler[:,2]).view(batch,1)#batch*1
    s3=torch.sin(euler[:,2]).view(batch,1)#batch*1

    # row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    # row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    # row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3

    # ZYX
    row1=torch.cat((c1*c2,  c1*s2*s3 - c3*s1,  s1*s3 + c1*c3*s2), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c2*s1,  c1*c3 + s1*s2*s3,  c3*s1*s2 - c1*s3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((-s2,    c2*s3,             c2*c3), 1).view(-1,1,3) #batch*1*3

    # row1=torch.cat((c2*c3,             -c2*s3,           s2), 1).view(-1,1,3) #batch*1*3
    # row2=torch.cat((c1*s3 + c3*s1*s2,  c1*c3 - s1*s2*s3,  -c2*s1), 1).view(-1,1,3) #batch*1*3
    # row3=torch.cat((s1*s3 - c1*c3*s2,  c3*s1 + c1*s2*s3, c1*c2), 1).view(-1,1,3) #batch*1*3

    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3


    return matrix

#r_matrix batch*3*3
def compute_ortho6d_from_rotation_matrix(r_matrix):
    ortho6d = torch.cat([r_matrix[:, :, 0], r_matrix[:, :, 1]], axis=1)
    return ortho6d

#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix