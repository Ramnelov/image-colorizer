import logging

from torch.optim import RAdam

from Evaluation import Evaluator
from Modules import UNet
from Training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


def main() -> None:

    trainer = Trainer(model=UNet(), batch_size=8, opt=RAdam, lr=1e-5)
    trained_model = trainer.train(num_epochs=10)

    evaluator = Evaluator(
        model=trained_model,
    )
    evaluator.evaluate(num_examples=10)


if __name__ == "__main__":
    main()
