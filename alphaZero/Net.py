import torch
import torch.nn as nn
import torch.optim as optim
import os

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        available = torch.cuda.is_available()
        self.device = torch.device('cuda' if available else 'cpu')
        print("Cuda available: ", available)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.to(self.device)  # Move the model to the device

    def forward(self, board):
        board = board.view(-1)  # Flatten the board
        x = torch.relu(self.fc1(board))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc3(x), dim=0)
        v = torch.tanh(self.fc4(x))
        return pi, v

    def train_model(self, examples):
        losses = []
        for epoch in range(10):
            self.train()
            epoch_loss = 0
            for board, pi, v in examples:
                board = torch.tensor(board, dtype=torch.float).view(-1).to(self.device)  # Flatten the board and move to device
                pi = torch.tensor(pi, dtype=torch.float).to(self.device)
                v = torch.tensor(v, dtype=torch.float).to(self.device)

                out_pi, out_v = self(board)
                loss_pi = -torch.sum(pi * torch.log(out_pi))
                loss_v = torch.sum((v - out_v) ** 2)
                total_loss = loss_pi + loss_v

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()

            avg_epoch_loss = epoch_loss / len(examples)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}")

        return losses

    def predict(self, board):
        self.eval()
        board = torch.tensor(board, dtype=torch.float).view(-1).to(self.device)  # Flatten the board and move to device
        pi, v = self(board)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()  # Move output back to CPU before converting to numpy

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        self.load_state_dict(torch.load(filepath))
        self.to(self.device)  # Ensure the model is moved to the device after loading
