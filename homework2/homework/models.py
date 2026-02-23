"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        loss = nn.CrossEntropyLoss()

        return loss(logits,target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super(LinearClassifier, self).__init__()
        c = 3 * h * w # flattened input length
        self.model = torch.nn.Sequential(
          torch.nn.Flatten(start_dim=1), # Flattens image to a vector 1st layer
          torch.nn.Linear(c,num_classes) # Linear layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super(MLPClassifier, self).__init__()
        c = 3 * h * w # flattened input length
        self.model = torch.nn.Sequential(
          torch.nn.Flatten(start_dim=1), # Flattens image to a vector 1st layer
          torch.nn.Linear(c,128), # Linear layer
          torch.nn.ReLU(), # Activation function
          torch.nn.Linear(128,num_classes) # Linear layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_lyrs: list = [192,128,64,64]
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super(MLPClassifierDeep, self).__init__()
        c = 3 * h * w # flattened input length
        # Create list for layers
        layers_ls = [torch.nn.Flatten(start_dim=1)] # Flattens image to a vector 1st layer
        
        # compile layers
        for lyr in hidden_lyrs:
          layers_ls.append(nn.Linear(c,lyr))
          layers_ls.append(nn.ReLU())
          c = lyr
        
        # Add layer to format output to correct size
        layers_ls.append(nn.Linear(c,num_classes))

        # Compile layers
        self.model = torch.nn.Sequential(*layers_ls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_lyrs: list = [192,128,64,64]
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        class Block(torch.nn.Module): # Define block of layers
          def __init__(self, in_channels,out_channels):
              super().__init__()
              self.conv = torch.nn.Linear(in_channels,out_channels)
              self.norm = torch.nn.LayerNorm(out_channels)
              self.relu = torch.nn.ReLU()
              # Check if skip connection is neccessary
              if in_channels != out_channels:
                  self.skip = torch.nn.Linear(in_channels,out_channels)
              else:
                  self.skip = torch.nn.Identity()

          def forward(self,x):
              y = self.relu(self.norm(self.conv(x)))
              return  y + self.skip(x)

        super(MLPClassifierDeepResidual, self).__init__()
        c = 3 * h * w # flattened input length
        # Create list for layers
        layers_ls = [torch.nn.Flatten(start_dim=1)] # Flattens image to a vector 1st layer
        # Create embedding layer
        embedding = hidden_lyrs.pop(0)
        layers_ls.append(torch.nn.Linear(c,embedding))
        c = embedding
        for s in layers_ls: # Go through each and add a layer
            layers_ls.append(self.Block(c,s)) # Add entire block at once
            c = s # save output dim as input for next layer

        # Add layer to format output to correct size
        layers_ls.append(nn.Linear(c,num_classes))

        # Compile layers
        self.model = torch.nn.Sequential(*layers_ls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)

model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
