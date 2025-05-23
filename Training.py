import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm import tqdm

nn = torch.nn
F = nn.functional


class Trainer:
    """Represents a trainer"""

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        opt: Optimizer,
        lr: float,
        train_ratio: float = 0.8,
    ) -> None:
        """Initialize the trainer"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).to(torch.float32)
        self.opt = opt(self.model.parameters(), lr=lr)
        self.batch_size = batch_size

        data = CIFAR100(root="./data", train=True, download=True, transform=ToTensor())
        train_size = int(train_ratio * len(data))
        val_size = len(data) - train_size

        self.train_data, self.val_data = torch.utils.data.random_split(
            data, [train_size, val_size]
        )

    def train(self, num_epochs: int) -> nn.Module:
        """Train the model"""

        train_loss_history = []
        val_loss_history = []
        for epoch in range(num_epochs):

            self.model.train()
            pbar = tqdm(
                DataLoader(
                    dataset=self.train_data, batch_size=self.batch_size, shuffle=True
                ),
                desc=f"Training Epoch {epoch+1}/{num_epochs}",
            )
            total_loss = 0
            total_samples = 0
            for x, _ in pbar:

                x = x.to(self.device).to(torch.float32)
                x_gray = rgb_to_grayscale(x, num_output_channels=1)

                # Predict output
                y_hat = self.model(x_gray)

                # Get loss and step
                loss = F.mse_loss(x, y_hat)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()
                total_samples += len(x)

                pbar.set_postfix(loss=loss.item() / len(x))

            train_loss_history.append(total_loss / total_samples)

            self.model.eval()
            pbar = tqdm(
                DataLoader(
                    dataset=self.val_data, batch_size=self.batch_size, shuffle=False
                ),
                desc=f"Validation Epoch {epoch+1}/{num_epochs}",
            )
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for x, _ in pbar:

                    x = x.to(self.device, dtype=torch.float32)
                    x_gray = rgb_to_grayscale(x, num_output_channels=1)

                    # Predict output
                    y_hat = self.model(x_gray)

                    # Get loss
                    loss = F.mse_loss(x, y_hat)

                    total_loss += loss.item()
                    total_samples += len(x)
                    pbar.set_postfix(avg_loss=total_loss / total_samples)

            val_loss_history.append(total_loss / total_samples)

        # Plot loss history
        epochs = range(1, len(train_loss_history) + 1)
        plt.plot(epochs, train_loss_history, label="Train Loss")
        plt.plot(epochs, val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Average Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

        return self.model
