import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Sample Data
torch.manual_seed(42)
nodes = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
labels = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
data = Data(x=nodes, edge_index=edges, y=labels)

# Model Initialization
model = GCN(input_dim=2, hidden_dim=4, output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Training
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = loss_fn(output, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
predictions = model(data.x, data.edge_index).detach().numpy()
print("Predictions:", predictions)
