import torch
import torch.nn as nn
import pytest
from unittest.mock import patch
from src.models.face_occ_regressor import FaceOccModel

class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 128
        self.num_prefix = 1
    def forward(self, x):
        # Return dummy tokens: (B, N, D)
        # 224/16 = 14. 14*14 = 196. Total 1+196 = 197 tokens.
        return torch.randn(x.shape[0], 197, 128)

@pytest.mark.parametrize("pooling_type", ["mean", "attention", "grid"])
def test_face_occ_model_forward(pooling_type):
    with patch("src.models.face_occ_regressor.build_backbone", return_value=MockBackbone()):
        model = FaceOccModel(
            backbone="mock", 
            pooling_type=pooling_type, 
            pretrained=False,
            grid_size=4
        )
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

def test_pooling_parameters():
    # Verify that poolings have parameters (for Weight Decay)
    with patch("src.models.face_occ_regressor.build_backbone", return_value=MockBackbone()):
        for p_type in ["mean", "attention", "grid"]:
            model = FaceOccModel(
                backbone="mock", 
                pooling_type=p_type, 
                pretrained=False
            )
            # Check if the pool module has parameters
            params = list(model.pool.parameters())
            assert len(params) > 0, f"Pooling {p_type} should have parameters for Weight Decay"
            
            # In our implementation:
            # MeanPool: self.proj
            # AttentionPool: self.q, self.proj
            # GridPool: self.proj
            if p_type == "attention":
                # Parameter q + Linear proj (weight + bias) = 3
                assert len(params) == 3
            elif p_type in ["mean", "grid"]:
                # Linear proj (weight + bias) = 2
                assert len(params) == 2
