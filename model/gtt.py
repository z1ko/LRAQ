import torch
import torch.nn as nn
import einops as ein

from model.temporal.lru import LRULayer
from model.spatial.gmlp import SGULayer

# Describes the kimore skeleton for semantic data
# https://www.researchgate.net/figure/Skeleton-provided-by-Microsoft-Kinect-v2-and-points-excluded-from-analysis-grey-ovals_fig4_332993109
kimore_skeleton_descriptor = {
    'joint_count': 17,
    'joint_features': 6,
    'joint_groups': 6,
    'joints': {
        'spine_base': {
            'id': 0,
            'connections': [1,10,13],
            'horz': 'center',
            'vert': 'low',
            'part': 0
        },
        'spine_mid': { 
            'id': 1,
            'connections': [0,16],
            'horz': 'center',
            'vert': 'low',
            'part': 0
        },
        'spine_shoulder': {
            'id': 16,
            'connections': [2,4,7,1],
            'horz': 'center',
            'vert': 'up',
            'part': 0
        },
        'neck': { 
            'id': 2,
            'connections': [3, 16],
            'horz': 'center',
            'vert': 'top',
            'part': 1
        },
        'head': {
            'id': 3,
            'connections': [2],
            'horz': 'center',
            'vert': 'top',
            'part': 1
        },
        'shoulder_left': {
            'id': 4,
            'connections': [5, 16],
            'horz': 'left',
            'vert': 'top',
            'part': 2
        },
        'elbow_left': {
            'id': 5,
            'connections': [4, 6],
            'horz': 'left',
            'vert': 'top',
            'part': 2
        },
        'wrist_left': { 
            'id': 6,
            'connections': [5],
            'horz': 'left',
            'vert': 'top',
            'part': 2
        },
        'shoulder_right': { 
            'id': 7,
            'connections': [8, 16],
            'horz': 'right',
            'vert': 'top',
            'part': 3
        },
        'elbow_right': {
            'id': 8,
            'connections': [7, 9],
            'horz': 'right',
            'vert': 'top',
            'part': 3
        },
        'wrist_right': {
            'id': 9,
            'connections': [8],
            'horz': 'right',
            'vert': 'top',
            'part': 3
        },
        'hip_left': {
            'id': 10,
            'connections': [0, 11],
            'horz': 'left',
            'vert': 'low',
            'part': 4
        },
        'knee_left': {
            'id': 11,
            'connections': [10,12],
            'horz': 'left',
            'vert': 'low',
            'part': 4
        },
        'ankle_left': {
            'id': 12,
            'connections': [11],
            'horz': 'left',
            'vert': 'low',
            'part': 4
        },
        'hip_right': {
            'id': 13,
            'connections': [0, 14],
            'horz': 'right',
            'vert': 'low',
            'part': 5
        },
        'knee_right': {
            'id': 14,
            'connections': [13,15],
            'horz': 'right',
            'vert': 'low',
            'part': 5
        },
        'ankle_right': {
            'id': 15,
            'connections': [14],
            'horz': 'right',
            'vert': 'low',
            'part': 5
        },
    }
}

# Spatial attention model with semantic augmentation
#class spatial_model(nn.Module):
#
#    _keywords = {
#        'left': 0, 
#        'right': 1, 
#        'center': 2,
#        'top': 0, 
#        'down': 1
#    }
#
#    def __init__(self, skeleton, layers):
#        super().__init__()
#
#        self.joint_count = skeleton['joint_count']
#        self.joint_features = skeleton['joint_features']
#        self.joint_groups = skeleton['joint_groups']
#
#        # Prepare onehot for joints, groups, vertical and horizontal placement.
#        self.create_semantic_features(self.joint_count, self.joint_groups)
#
#        features_dim = self.joint_count + 3 + 2 + self.joint_groups
#        self.semantic = torch.zeros((self.joint_count, self.joint_count + features_dim))
#        for _, joint in skeleton['joints']:
#            
#            idx = joint['id']
#            self.semantic[idx, 0:self.joint_count] = self.onehot_joints(idx)
#            self.semantic[idx, self.joint_count:self.joint_count+3] = self.onehot_horz(self._keywords[joint['horz']])
#            self.semantic[idx, self.joint_count+3:self.joint_count+5]= self.onehot_vert(self._keywords[joint['vert']])
#            self.semantic[idx, self.joint_count+5:] = self.onehot_groups[joint['group']]
#
#        # Spatial layers
#        self.layers = nn.ModuleList([ spatial_gat() for _ in range(layers) ])
#
#    def create_semantic_features(self, joint_count, joint_groups):
#        self.onehot_joints = torch.eye(joint_count)
#        self.onehot_groups = torch.eye(joint_groups)
#        self.onehot_horz = torch.eye(3)
#        self.onehot_vert = torch.eye(2)
#
#    def forward(self, x): # B T J D
#        x = torch.cat([x, self.semantic], dim=-1)
#        for layer in self.layers:
#            x = layer(x)
#        return x


# A Graph-to-Time model
class G2TAQ(nn.Module):
    def __init__(
        self,
        model_dim,
        joint_count,
        joint_feaures,
        temporal_dim,
        spatial_dim,
        **kwargs
    ):
        super().__init__()

        # initial embedding
        self.embed = nn.Linear(joint_feaures, model_dim)

        self.spatial = SGULayer(model_dim, spatial_dim, joint_count, **kwargs)
        self.spatial_agg = nn.Sequential(
            nn.Linear(model_dim * joint_count, model_dim * 5),
            nn.GELU(),
            nn.Linear(model_dim * 5, model_dim)
        )
        
        self.temporal = LRULayer(model_dim, temporal_dim, 5, True, **kwargs)
        self.regression = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x): # (B, T, J, D)

        # Spatial reasoning
        x = self.embed(x)
        x = self.spatial(x)

        # Temporal reasoning
        B, T, J, D = x.shape
        x = ein.rearrange(x, 'B T J D -> (B J) T D')
        x = self.temporal(x)

        # Final spatial and temporal aggregation
        x = ein.rearrange(x, '(B J) T D -> B T (J D)', B=B, J=J)
        x = self.spatial_agg(x)
        x = torch.mean(x, dim=1)
        x = self.regression(x)
        return x.squeeze()


        
