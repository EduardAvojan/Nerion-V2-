import torch
from nerion_digital_physicist.agent.brain import CodeGraphNN

def test_forward_with_dropout():
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
