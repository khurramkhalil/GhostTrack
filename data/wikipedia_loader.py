"""Wikipedia corpus loader for SAE training."""

import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, List
from datasets import load_dataset
from tqdm import tqdm


class WikipediaCorpus:
    """
    Loader for Wikipedia text corpus.

    Used for extracting hidden states to train SAEs.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        language: str = 'en',
        date: str = '20220301',
        max_tokens: int = 100_000_000
    ):
        """
        Initialize Wikipedia corpus loader.

        Args:
            cache_dir: Directory to cache dataset.
            language: Wikipedia language (default: 'en').
            date: Wikipedia dump date (default: '20220301').
            max_tokens: Maximum number of tokens to process.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data/cache')
        self.language = language
        self.date = date
        self.max_tokens = max_tokens
        self.dataset = None
        self.streaming = True
        # Auto-load dataset
        self.load()

    def load(self):
        """Load Wikipedia dataset from HuggingFace."""
        print(f"Loading Wikipedia dataset ({self.language}, {self.date})...")

        self.dataset = load_dataset(
            'wikipedia',
            f'{self.date}.{self.language}',
            split='train',
            cache_dir=str(self.cache_dir),
            streaming=True  # Use streaming to avoid loading entire dataset
        )

        print("Wikipedia dataset loaded (streaming mode)")
        return self

    def get_texts(self, max_texts: Optional[int] = None) -> Iterator[str]:
        """
        Get text iterator from Wikipedia.

        Args:
            max_texts: Maximum number of texts to yield.

        Yields:
            Wikipedia article texts.
        """
        if self.dataset is None:
            self.load()

        count = 0
        for item in self.dataset:
            if max_texts and count >= max_texts:
                break

            text = item['text']
            if len(text.strip()) > 100:  # Filter out very short articles
                yield text
                count += 1

    def get_batch(self, batch_size: int, min_length: int = 0) -> List[str]:
        """
        Get a single batch of texts.

        Args:
            batch_size: Number of texts to return.
            min_length: Minimum text length filter.

        Returns:
            List of Wikipedia texts.
        """
        if batch_size == 0:
            return []

        batch = []
        for text in self.get_texts():
            if len(text) >= min_length:
                batch.append(text)
            if len(batch) >= batch_size:
                break
        return batch

    def get_text_batches(
        self,
        batch_size: int = 32,
        max_batches: Optional[int] = None
    ) -> Iterator[List[str]]:
        """
        Get batches of texts.

        Args:
            batch_size: Number of texts per batch.
            max_batches: Maximum number of batches to yield.

        Yields:
            Batches of Wikipedia texts.
        """
        batch = []
        batch_count = 0

        for text in self.get_texts():
            batch.append(text)

            if len(batch) >= batch_size:
                yield batch
                batch = []
                batch_count += 1

                if max_batches and batch_count >= max_batches:
                    break

        # Yield remaining texts
        if batch:
            yield batch


class HiddenStateExtractor:
    """
    Extract hidden states from model on text corpus.
    """

    def __init__(
        self,
        model_wrapper,
        corpus: WikipediaCorpus,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize hidden state extractor.

        Args:
            model_wrapper: GPT2WithResidualHooks instance.
            corpus: WikipediaCorpus instance.
            cache_dir: Directory to cache extracted states.
            device: Device to run model on.
        """
        self.model = model_wrapper
        self.corpus = corpus
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data/cache/hidden_states')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def extract_from_text(
        self,
        text: str,
        layer_idx: int,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Extract hidden states from a single text.

        Args:
            text: Input text.
            layer_idx: Layer index to extract from.
            max_length: Maximum sequence length.

        Returns:
            Hidden states as numpy array [seq_len, d_model].
        """
        import torch
        import numpy as np

        with torch.no_grad():
            outputs = self.model.process_text(text, max_length=max_length)
            hidden_states = outputs['residual_stream'][layer_idx]  # [1, seq_len, d_model]
            return hidden_states[0].cpu().numpy()  # [seq_len, d_model]

    def extract_from_batch(
        self,
        texts: List[str],
        layer_idx: int,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Extract hidden states from multiple texts.

        Args:
            texts: List of input texts.
            layer_idx: Layer index to extract from.
            max_length: Maximum sequence length.

        Returns:
            Concatenated hidden states as numpy array [total_tokens, d_model].
        """
        all_states = []

        for text in texts:
            states = self.extract_from_text(text, layer_idx, max_length)
            all_states.append(states)

        return np.concatenate(all_states, axis=0)

    def extract_for_layer(
        self,
        layer_idx: int,
        num_tokens: int = 10_000_000,
        batch_size: int = 8,
        max_length: int = 512,
        save_every: int = 100,
        save_dir: Optional[str] = None
    ) -> Path:
        """
        Extract hidden states for a specific layer.

        Args:
            layer_idx: Layer index (0-11 for GPT-2).
            num_tokens: Target number of tokens to collect.
            batch_size: Batch size for processing.
            max_length: Maximum sequence length.
            save_every: Save checkpoint every N batches.
            save_dir: Optional directory to save to (overrides cache_dir).

        Returns:
            Path to saved hidden states file.
        """
        output_dir = Path(save_dir) if save_dir else self.cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'layer_{layer_idx}_states.pt'

        # Check if already exists
        if output_file.exists():
            print(f"Hidden states for layer {layer_idx} already exist at {output_file}")
            return output_file

        print(f"\nExtracting hidden states for layer {layer_idx}...")
        print(f"Target: {num_tokens:,} tokens")

        all_states = []
        total_tokens = 0
        batch_count = 0

        # Calculate approximate batches needed
        avg_tokens_per_batch = batch_size * max_length * 0.7  # Assume 70% utilization
        approx_batches = int(num_tokens / avg_tokens_per_batch) + 1

        pbar = tqdm(total=num_tokens, desc=f"Layer {layer_idx}", unit="tokens")

        try:
            for texts in self.corpus.get_text_batches(batch_size=batch_size):
                # Tokenize batch
                encoded = self.model.tokenizer(
                    texts,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )

                input_ids = encoded['input_ids'].to(self.model.device)
                attention_mask = encoded['attention_mask'].to(self.model.device)

                # Forward pass
                outputs = self.model.forward_with_cache(input_ids, attention_mask)

                # Extract layer activations
                layer_states = outputs['residual_stream'][layer_idx]  # [batch, seq, d_model]

                # Flatten to [num_tokens, d_model] and filter padding
                for i in range(layer_states.shape[0]):
                    # Get actual tokens (non-padding)
                    actual_length = attention_mask[i].sum().item()
                    states = layer_states[i, :actual_length, :].cpu()
                    all_states.append(states)
                    total_tokens += states.shape[0]

                pbar.update(states.shape[0] * layer_states.shape[0])
                batch_count += 1

                # Save checkpoint
                if batch_count % save_every == 0:
                    self._save_checkpoint(all_states, output_file, layer_idx)

                # Check if we have enough tokens
                if total_tokens >= num_tokens:
                    break

        finally:
            pbar.close()

        # Final save
        print(f"\nCollected {total_tokens:,} tokens for layer {layer_idx}")
        final_path = self._save_final(all_states, output_file, layer_idx)

        return final_path

    def _save_checkpoint(self, states: List[torch.Tensor], output_file: Path, layer_idx: int):
        """Save intermediate checkpoint."""
        checkpoint_file = self.cache_dir / f'layer_{layer_idx}_checkpoint.pt'
        concatenated = torch.cat(states, dim=0)
        torch.save({
            'hidden_states': concatenated,
            'layer_idx': layer_idx,
            'num_tokens': concatenated.shape[0]
        }, checkpoint_file)

    def _save_final(
        self,
        states: List[torch.Tensor],
        output_file: Path,
        layer_idx: int
    ) -> Path:
        """Save final hidden states."""
        # Concatenate all states
        concatenated = torch.cat(states, dim=0)

        print(f"Saving {concatenated.shape[0]:,} tokens to {output_file}")

        torch.save({
            'hidden_states': concatenated,
            'layer_idx': layer_idx,
            'num_tokens': concatenated.shape[0],
            'd_model': concatenated.shape[1]
        }, output_file)

        # Remove checkpoint if exists
        checkpoint_file = self.cache_dir / f'layer_{layer_idx}_checkpoint.pt'
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        return output_file

    def extract_all_layers(
        self,
        num_tokens: int = 10_000_000,
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[Path]:
        """
        Extract hidden states for all layers.

        Args:
            num_tokens: Target tokens per layer.
            batch_size: Batch size.
            max_length: Max sequence length.

        Returns:
            List of paths to saved states for each layer.
        """
        paths = []

        for layer_idx in range(self.model.n_layers):
            path = self.extract_for_layer(
                layer_idx=layer_idx,
                num_tokens=num_tokens,
                batch_size=batch_size,
                max_length=max_length
            )
            paths.append(path)

        return paths


def load_hidden_states(file_path: str) -> torch.Tensor:
    """
    Load hidden states from file.

    Args:
        file_path: Path to saved states.

    Returns:
        Hidden states tensor [num_tokens, d_model].
    """
    data = torch.load(file_path)
    return data['hidden_states']
