"""
Changed to lightning by Belter, Jan 13, 2025.
"""

from typing import Any, Optional
import torch
import lightning as L


class BaseEncoder(L.LightningModule):
    """Base class for encoder neural networks in VAE architectures."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, knn_edge_index: Optional[Any], ppi_edge_index: Optional[Any]) -> Any:
        """Forward pass of the encoder.

        This method must be implemented in child classes. It processes the input data
        and returns an encoded representation.

        Args:
            x (torch.Tensor): Input data to be encoded
            knn_edge_index (Optional[torch.Tensor]): KNN edge index, only for GNN
            ppi_edge_index (Optional[torch.Tensor]): PPI edge index, only for GNN

        Returns:
            ModelOutput: Encoded representation of the input

        Raises:
            NotImplementedError: If not implemented in child class
        """
        raise NotImplementedError("Forward method must be implemented in child class")


class BaseDecoder(L.LightningModule):
    """Base class for decoder neural networks in VAE architectures."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Any:
        """Forward pass of the decoder.

        This method must be implemented in child classes. It processes the latent
        representation and returns the reconstructed data.

        Args:
            z (torch.Tensor): Latent representation to be decoded

        Returns:
            ModelOutput: Reconstructed data

        Note:
            Reconstruction tensors should be in range [0, 1] with shape:
            (batch_size, channels, ...)

        Raises:
            NotImplementedError: If not implemented in child class
        """
        raise NotImplementedError("Forward method must be implemented in child class")


class BaseMetric(L.LightningModule):
    """Base class for metric neural networks in Riemannian VAE architectures."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass of the metric network.

        This method must be implemented in child classes. It computes the
        Riemannian metric for the input data.

        Args:
            x (torch.Tensor): Input data for metric computation

        Returns:
            ModelOutput: Computed metric values

        Raises:
            NotImplementedError: If not implemented in child class
        """
        raise NotImplementedError("Forward method must be implemented in child class")


class BaseDiscriminator(L.LightningModule):
    """Base class for discriminator neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass of the discriminator.

        This method must be implemented in child classes. It processes the input
        data and returns discrimination results.

        Args:
            x (torch.Tensor): Input data to be discriminated

        Returns:
            ModelOutput: Discrimination results

        Raises:
            NotImplementedError: If not implemented in child class
        """
        raise NotImplementedError("Forward method must be implemented in child class")
