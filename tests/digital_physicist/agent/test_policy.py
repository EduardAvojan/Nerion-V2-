import torch
from torch_geometric.data import Data
from nerion_digital_physicist.agent.policy import AgentV2
from nerion_digital_physicist.environment.refactor_env import RenameAction

def test_predict_with_uncertainty():
    agent = AgentV2(input_dim=11, hidden_dim=16, num_mc_passes=10)
    
    graph_data = Data(
        x=torch.randn(5, 10),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
    )
    
    node_map = {
        0: {"id": "file.py::foo", "name": "foo"},
        1: {"id": "file.py::bar", "name": "bar"},
    }
    
    action = RenameAction(file_path="file.py", old_name="foo", new_name="baz")
    
    mean, uncertainty = agent.predict_with_uncertainty(graph_data, node_map, action)
    
    assert isinstance(mean, torch.Tensor)
    assert isinstance(uncertainty, torch.Tensor)
    assert uncertainty.item() >= 0
