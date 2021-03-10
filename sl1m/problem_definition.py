from sl1m.constants_and_tools import *

from . import qp

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

#LP contact planner using inequality formulation


############### Problem definition #############


LF = 0
RF = 1

def normalize(Ab):
    A = Ab[0]
    b = Ab[1]
    Ares = zeros(A.shape)
    bres = zeros(b.shape)
    for i in range(A.shape[0]):
        n = norm(A[i,:])
        if n <= 0.000001:
            n = 1.
        Ares[i,:] = A[i,:] / n; bres[i] = b[i] / n
    return Ares, bres


def genKinematicConstraints(lf_constraints_fun, rf_constraints_fun, index = 0, rotation = [Id,Id], normals = [z, z], min_height = None):   #assume that root transform is given in 3x3 rotation matrix
    res = [None, None]
    if index == 0 :
        trLF = default_transform_from_pos_normal(zero3, normals[LF], rotation[index])
        trRF = default_transform_from_pos_normal(zero3, normals[RF], rotation[index])
    elif index % 2 == LF : # left foot is moving
        trLF = default_transform_from_pos_normal(zero3, normals[LF], rotation[index])
        trRF = default_transform_from_pos_normal(zero3, normals[RF], rotation[index-1])
    elif index % 2 == RF : # right foot is moving
        #print index
        trLF = default_transform_from_pos_normal(zero3, normals[LF], rotation[index-1])
        trRF = default_transform_from_pos_normal(zero3, normals[RF], rotation[index])

    #~ KLF = left_foot_talos_constraints  (trLF)
    #~ KRF = right_foot_talos_constraints (trRF)
    #~ KLF = left_foot_hrp2_constraints (trLF)
    #~ KRF = right_foot_hrp2_constraints  (trRF)
    KLF = lf_constraints_fun (trLF)
    KRF = rf_constraints_fun  (trRF)
    if min_height is None:
        res [LF] = KLF
        res [RF] = KRF
    else:
        res [LF] = addHeightConstraint(KLF[0], KLF[1], min_height)
        res [RF] = addHeightConstraint(KRF[0], KRF[1], min_height)
    return res
     
def genFootRelativeConstraints(rf_in_lf_frame_constraints_fun,  lf_in_rf_frame_constraints_fun, index = 0, rotation = [Id,Id], normals = [z, z]): #assume that root transform is given in 3x3 rotation matrix
    res = [None, None]
    if index == 0 :
        trLF = default_transform_from_pos_normal(zero3, normals[LF], rotation[index])
        trRF = default_transform_from_pos_normal(zero3, normals[RF], rotation[index])
    elif index % 2 == LF : # left foot is moving
        trLF = default_transform_from_pos_normal(zero3, normals[LF], rotation[index])
        trRF = default_transform_from_pos_normal(zero3, normals[RF], rotation[index-1])
    elif index % 2 == RF : # right foot is moving
        trLF = default_transform_from_pos_normal(zero3, normals[LF], rotation[index-1])
        trRF = default_transform_from_pos_normal(zero3, normals[RF], rotation[index])
    #~ KRF = right_foot_in_lf_frame_talos_constraints  (trLF)
    #~ KLF = left_foot_in_rf_frame_talos_constraints (trRF)   
    
    #~ KRF = right_foot_in_lf_frame_hrp2_constraints  (trLF)
    #~ KLF = left_foot_in_rf_frame_hrp2_constraints (trRF)    
     
    KRF = rf_in_lf_frame_constraints_fun  (trLF)
    KLF = lf_in_rf_frame_constraints_fun (trRF)     
    res [LF] = KLF #constraints of right foot in lf frame. Same idea as COM in lf frame
    res [RF] = KRF
    return res
    
    
def copyKin(kC):
    return [(Kk[0].copy(), Kk[1].copy()) for Kk in kC]
