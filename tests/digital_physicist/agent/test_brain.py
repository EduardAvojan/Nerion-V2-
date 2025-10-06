import torch
import pytest
from nerion_digital_physicist.agent.brain import CodeGraphNN


def test_forward_with_dropout():
    """Test that dropout produces different outputs on multiple calls."""
    model = CodeGraphNN(num_node_features=10, hidden_channels=16, num_classes=2)
    model.eval()  # Set the model to evaluation mode

    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)

    # With use_dropout=True, the output should be different on multiple calls
    output1 = model(x, edge_index, batch, use_dropout=True)
    output2 = model(x, edge_index, batch, use_dropout=True)

    assert not torch.allclose(output1, output2)

    # Without use_dropout=True, the output should be the same
    output3 = model(x, edge_index, batch)
    output4 = model(x, edge_index, batch)

    assert torch.allclose(output3, output4)


def test_model_initialization():
    """Test model initialization with different parameters."""
    # Test default initialization
    model1 = CodeGraphNN(num_node_features=5, hidden_channels=8, num_classes=2)
    assert model1.num_node_features == 5
    assert model1.hidden_channels == 8
    assert model1.num_classes == 2
    
    # Test with different parameters
    model2 = CodeGraphNN(num_node_features=20, hidden_channels=32, num_classes=3)
    assert model2.num_node_features == 20
    assert model2.hidden_channels == 32
    assert model2.num_classes == 3


def test_model_forward_pass():
    """Test forward pass with various input sizes."""
    model = CodeGraphNN(num_node_features=10, hidden_channels=16, num_classes=2)
    model.eval()
    
    # Test with different graph sizes
    for num_nodes in [3, 5, 10]:
        x = torch.randn(num_nodes, 10)
        edge_index = torch.tensor([[i for i in range(num_nodes)], 
                                 [(i + 1) % num_nodes for i in range(num_nodes)]], 
                                dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        output = model(x, edge_index, batch)
        assert output.shape == (num_nodes, 2)


def test_model_training_mode():
    """Test model behavior in training vs evaluation mode."""
    model = CodeGraphNN(num_node_features=10, hidden_channels=16, num_classes=2)
    
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    
    # In training mode, dropout should be active
    model.train()
    output_train = model(x, edge_index, batch)
    
    # In eval mode, dropout should be inactive
    model.eval()
    output_eval = model(x, edge_index, batch)
    
    # Outputs should be different due to dropout in training mode
    assert not torch.allclose(output_train, output_eval)
