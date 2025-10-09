import pennylane as qml
import torch
import torch.nn as nn

class HybridQuantumClassifier(nn.Module):
    def __init__(self, num_qubits: int, num_layers: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Lightning device with adjoint differentiation (fast and PyTorch compatible)
        self.dev = qml.device("lightning.qubit", wires=num_qubits)

        # QNode definition
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(0.01 * torch.randn(num_layers, num_qubits, 3))
        self.fc = nn.Linear(num_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch execution of QNode
        q_outs = torch.stack([
            torch.tensor(self.qnode(x[i], self.weights), dtype=x.dtype)
            for i in range(x.shape[0])
        ])
        return self.fc(q_outs).squeeze(1)


def train_qnn(model, X_train, y_train, epochs=5, batch_size=16, lr=0.01):
    """Train the QNN with batch processing and BCEWithLogitsLoss"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss: {running_loss/len(loader):.4f}")

    return model
