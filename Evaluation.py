import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale

nn = torch.nn
F = nn.functional


class Evaluator:
    """Represents an evaluator"""

    def __init__(self, model: nn.Module) -> None:
        """Initialize the evaluator"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).to(torch.float32)

        self.test_data = CIFAR100(
            root="./data", train=False, download=True, transform=ToTensor()
        )

    def evaluate(self, num_examples) -> None:
        """Evaluate model"""

        self.model.eval()

        # Get examples to evaluate
        x, _ = next(
            iter(DataLoader(self.test_data, batch_size=num_examples, shuffle=True))
        )
        x = x.to(self.device).to(torch.float32)
        x_gray = rgb_to_grayscale(x, num_output_channels=1)

        # Predict output
        with torch.no_grad():
            y_hat = self.model(x_gray)

        x = x.cpu()
        y_hat = y_hat.cpu()

        # Plot examples
        for i in range(num_examples):

            _, axes = plt.subplots(1, 2, figsize=(8, 4))

            target_img = np.transpose(x[i].numpy(), (1, 2, 0))
            predicted_img = np.transpose(y_hat[i].numpy(), (1, 2, 0))

            axes[0].imshow(target_img)
            axes[0].set_title("Target Image")
            axes[0].axis("off")

            axes[1].imshow(predicted_img)
            axes[1].set_title("Predicted Image")
            axes[1].axis("off")

            plt.show()
