import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm

'''todo: 
    1. make last layer linear instead of mlp
    2. alter normalization and see the output
    3. 
'''

# def cosine_distance(x1, x2=None, eps=1e-8):
#     x2 = x1 if x2 is None else x2
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
#     return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    

# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - a MLP that transforms this concatenated descriptor into the output 32-dimensional descriptor x_i
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()
        self.mlp1 = MLP([num_input_features, 32, 64, 128])
        self.mlp2 = MLP([128, 128])
        self.mlp3 = MLP([256, 128, 64])
        self.lin1 = torch.nn.Linear(64, num_output_features)
        
    def forward(self, x):

        x = self.mlp1(x)
        fi = x
        
        x = self.mlp2(x)
        
        x[:] = torch.max(x,0).values
        x = torch.cat((x, fi), 1)
        
        x = self.mlp3(x)
        
        x = self.lin1(x)

        return x


# CorrNet module that serves 2 purposes:  
# (a) uses the PointNet module to extract the per-point descriptors of the point cloud (out_pts)
#     and the same PointNet module to extract the per-vertex descriptors of the mesh (out_vtx)
# (b) if self.train_corrmask=1, it outputs a correspondence mask
# The CorrNet module should
# - include a (shared) PointNet to extract the per-point and per-vertex descriptors 
# - normalize these descriptors to have length one
# - when train_corrmask=1, it should include a MLP that outputs a confidence 
#   that represents whether the mesh vertex i has a correspondence or not
#   Specifically, you should use the cosine similarity to compute a similarity matrix NxM where
#   N is the number of mesh vertices, M is the number of points in the point cloud
#   Each entry encodes the similarity of vertex i with point j
#   Use the similarity matrix to find for each mesh vertex i, its most similar point n[i] in the point cloud 
#   Form a descriptor matrix X = NxF whose each row stores the point descriptor of n[i] (from the point cloud descriptors)
#   Form a vector S = Nx1 whose each entry stores the similarity of the pair (i, n[i])
#   From the PointNet, you also have the descriptor matrix Y = NxF storing the per-vertex descriptors
#   Concatenate [X Y S] into a N x (2F + 1) matrix
#   Transform this matrix into the correspondence mask Nx1 through a MLP followed by a linear transformation
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class CorrNet(torch.nn.Module):
    def __init__(self, num_output_features, train_corrmask):        
        super(CorrNet, self).__init__()
        self.train_corrmask = train_corrmask
        self.pointnet_share = PointNet(3, num_output_features)
        self.mlp = MLP([2*num_output_features+1, 64]) # you won't use this, delete it, this is there just for the code to run
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, vtx, pts):
        out_vtx = self.pointnet_share(vtx)
        out_pts = self.pointnet_share(pts)
        out_vtx = torch.nn.functional.normalize(out_vtx)
        out_pts = torch.nn.functional.normalize(out_pts)
        if self.train_corrmask:
            
            similarity_matrx = torch.matmul(out_vtx, out_pts.T)
            S, ni  = torch.max(similarity_matrx, 1)
            S = torch.reshape(S, (S.shape[0], 1))
            X = out_vtx
            X[:] = out_pts[ni[:]]
            output = torch.cat((out_vtx, X, S), 1)            
            output = self.mlp(output)          
            out_corrmask = self.lin(output)  

        else:
            out_corrmask = None
            

        return out_vtx, out_pts, out_corrmask
