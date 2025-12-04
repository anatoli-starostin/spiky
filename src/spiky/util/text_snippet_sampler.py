from typing import Iterator
import random

import torch


class TextSnippetSampler:
    """
    Sampler for text snippets with training/testing split.

    Loads text data, marks testing regions, and provides methods to sample
    training batches and iterate over testing batches.
    """

    def __init__(
        self,
        text_file_name: str,
        context_size: int,
        n_test_regions: int,
        device: torch.device,
        random_seed: int = None,
    ):
        """
        Initialize the text snippets sampler.

        Args:
            text_file_name: Path to text file to load
            context_size: Size of context window
            n_test_regions: Number of testing regions to mark
            device: Device to store tensors on
            random_seed: Optional random seed for reproducible region sampling
        """
        self.context_size = context_size
        self.n_test_regions = n_test_regions
        self.device = device

        # Load text file as bytes and convert to int32 tensor
        with open(text_file_name, "rb") as f:
            text_bytes = f.read()

        # Convert bytes to int32 tensor (each byte becomes an int in 0-255)
        self.text = torch.tensor(list(text_bytes), dtype=torch.int32, device=device)
        self.text_length = len(self.text)

        # Create testing mask: mark regions of size 2 * context_size as testing
        self.testing_mask = torch.zeros(self.text_length, dtype=torch.bool, device=torch.device('cpu'))

        # Sample random centers for test regions
        # Each test region is 2 * context_size in size
        test_region_size = 2 * context_size
        half_region = test_region_size // 2
        min_center = half_region
        max_center = self.text_length - half_region

        # Create an optional generator for reproducibility
        if random_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(random_seed)
        else:
            gen = None

        # Centers are sampled uniformly in [min_center, max_center] (inclusive)
        self.test_region_centers_tensor = torch.randint(
            low=min_center,
            high=max_center + 1,
            size=(n_test_regions,),
            dtype=torch.int64,
            device=device,
            generator=gen,
        )
        self.test_region_centers = self.test_region_centers_tensor.tolist()

        # Mark regions around centers as testing
        for center in self.test_region_centers:
            start = max(0, center - half_region)
            end = min(self.text_length, center + half_region)
            self.testing_mask[start:end] = True

    def sample_training_batch(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of training snippets.

        Samples random start positions that are not in testing regions.
        Since testing regions are 2x bigger than context_size, we can safely
        sample any position that's not marked as testing.

        Args:
            batch_size: Number of snippets to sample

        Returns:
            Tensor of shape (batch_size, context_size + 1) containing token indices.
            The last column is the target token for each snippet.
        """
        batch = []
        attempts = 0
        max_attempts = batch_size * 100  # Prevent infinite loops

        while len(batch) < batch_size and attempts < max_attempts:
            # Sample random start position
            start = random.randint(0, self.text_length - self.context_size - 1)

            if not self.testing_mask[start].item():
                batch.append(self.text[start:start + self.context_size])

            attempts += 1

        if len(batch) < batch_size:
            raise RuntimeError(
                f"Could not sample {batch_size} training snippets. "
                f"Only found {len(batch)} valid snippets after {attempts} attempts."
            )

        # Stack into batch tensor: (batch_size, context_size + 1)
        return torch.stack(batch, dim=0)

    def testing_batches_iterator(self, batch_size: int) -> Iterator[torch.Tensor]:
        """
        Iterator over all testing snippets grouped into batches.

        Each testing snippet has length `context_size` and starts at the center of
        its corresponding testing region.

        Args:
            batch_size: Size of each batch

        Yields:
            Tensors of shape (batch_size, context_size) containing token indices.
            Last batch may be smaller than batch_size.
        """
        # Collect all testing snippets
        test_snippets = []

        for center in self.test_region_centers:
            # Testing snippet of length context_size starting at the center
            start = center
            end = start + self.context_size

            # Ensure valid range
            if start >= 0 and end <= self.text_length:
                snippet = self.text[start:end]
                test_snippets.append(snippet)

        # Yield batches
        for i in range(0, len(test_snippets), batch_size):
            batch = test_snippets[i:i + batch_size]
            yield torch.stack(batch, dim=0)

    def batch_to_text(self, batch_of_ints):
        assert batch_of_ints.shape[1] == self.context_size
        results = []
        for i in range(batch_of_ints.shape[0]):
            results.append(''.join(chr(int(x)) for x in batch_of_ints[i].tolist()))
        return results
