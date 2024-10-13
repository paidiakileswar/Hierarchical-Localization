import sys
import torch
from pathlib import Path

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from SphereGlue.model.sphereglue import SphereGlue as SG


class SphereGlue(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 200,
        'match_threshold': 0.1,
        'descriptor_dim' : 256,
        'output_dim' : 256 * 2,
        'K': 2,
        'GNN_layers': ['cross'],
        'aggr': 'add',
        'knn': 20,
    }
    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = SG(conf).to(device)

    def _forward(self, data):
        return self.net(data)
