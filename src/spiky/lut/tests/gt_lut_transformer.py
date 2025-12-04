"""
Pure Python implementation of text_lut.c
SNN transformer architecture using LUT layers instead of attention.
"""

import math
import random
from typing import List, Tuple


# Constants matching text_lut.c
CONTEXT_SIZE = 8
VOCAB_SIZE = 256
EMBEDDING_DIM = 32
POSITIONAL_DIM = 4
NUM_LAYERS = 2
NUM_HEADS = 2
N_T = 8  # Number of detectors
N_C = 6  # Number of anchor pairs per detector
TESTING_LENGTH = 512


def sign(x: float) -> int:
    """Return sign of x. Zero has 'minus' sign."""
    return 1 if x > 0 else -1


def up(x: float) -> float:
    """Gradient function for backward pass."""
    return 0.5 * sign(x) / (1 + abs(x)) / (1 + abs(x))


def learning_rate_scheduler(step: int) -> float:
    """Adam-like learning rate scheduler: MIN(1/sqrt(1+t), t/(4000)/sqrt(4000))"""
    t = step + 1
    lr1 = 1.0 / math.sqrt(1 + t)
    lr2 = t / (4000.0 * math.sqrt(4000.0))
    return min(lr1, lr2)


def softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax with temperature scaling."""
    # Find max value for numerical stability
    # max_val = max(x)
    max_val = 0.0
    
    # Compute exp and sum
    exp_vals = [math.exp((val - max_val) / temperature) for val in x]
    sum_exp = sum(exp_vals)
    
    # Normalize
    return [val / sum_exp for val in exp_vals]


def sample(probabilities: List[float]) -> int:
    """Sample index from probabilities (they must sum to 1!)."""
    coin = random.random()
    cdf = 0.0
    for i, prob in enumerate(probabilities):
        cdf += prob
        if coin < cdf:
            return i
    return len(probabilities) - 1  # In case of rounding errors


class LUT:
    """
    LUT (Look-Up Table) layer combining LUT and LUTcache structures.
    """
    
    def __init__(self, total_n_c: int, y_dim: int, n_t: int = N_T, n_c: int = N_C, embedding_dim: int = EMBEDDING_DIM):
        """
        Initialize LUT.
        
        Args:
            total_n_c: Total number of anchor pairs (for table size calculation)
            y_dim: Output dimension
            n_t: Number of detectors (tables)
            n_c: Number of anchor pairs per detector
            embedding_dim: Input embedding dimension (for anchor initialization)
        """
        self.y_dim = y_dim
        self.n_t = n_t
        self.n_c = n_c
        self.embedding_dim = embedding_dim
        
        # Initialize anchors: each detector has n_c anchor pairs
        self.anchors = []
        for i in range(n_t):
            a = [random.randint(0, embedding_dim - 1) for _ in range(n_c)]
            b = []
            for j in range(n_c):
                while True:
                    b_val = random.randint(0, embedding_dim - 1)
                    if b_val != a[j]:
                        break
                b.append(b_val)
            self.anchors.append({'a': a, 'b': b})
        
        # Initialize synaptic values: S[i] is a table of size (2^total_n_c) * y_dim
        table_size = 1 << total_n_c
        self.S = []
        for i in range(n_t):
            self.S.append([0.0] * (table_size * y_dim))
        
        # Internal cache storage: dict[key] -> cache dict
        self.caches = {}
        # Accumulator for weight gradients: dict[(table_index, weight_idx)] -> grad
        self.v_weight_grads = {}
    
    def cache_index(self, x: List[float], key: int = 0) -> None:
        """
        Cache lookup indices for input x and store internally.
        
        Args:
            x: Input vector
            key: Key to identify this cache (e.g., position index)
        """
        cache = {
            'j': [0] * self.n_t,  # Lookup indices
            'r_min': [0] * self.n_t,  # Index of minimum delta anchor
            'u_min': [float('inf')] * self.n_t  # Minimum delta value
        }
        
        for i in range(self.n_t):
            cache['j'][i] = 0
            cache['u_min'][i] = float('inf')
            
            for r in range(self.n_c):
                u = x[self.anchors[i]['a'][r]] - x[self.anchors[i]['b'][r]]
                if u > 0:
                    cache['j'][i] |= (1 << r)  # Bit concatenation
                if abs(u) < abs(cache['u_min'][i]):
                    cache['r_min'][i] = r
                    cache['u_min'][i] = u
        
        # print(f'cached lut info {cache} for position {key}')
        self.caches[key] = cache

    def forward(self, y: List[float], key: int = 0) -> None:
        """
        Forward pass: accumulate outputs from LUT tables.
        
        Args:
            y: Output vector (will be modified in-place)
            key: Key to identify which cache to use
        """
        cache = self.caches[key]
        for i in range(self.n_t):
            j = cache['j'][i]
            for k in range(self.y_dim):
                y[k] += self.S[i][j * self.y_dim + k]
    
    def backward(self, x_gradient: List[float], y_gradient: List[float], learning_rate: float, key: int = 0) -> None:
        """
        Backward pass: compute gradients and update weights.
        
        Args:
            x_gradient: Input gradient (will be modified in-place)
            y_gradient: Output gradient
            learning_rate: Learning rate for weight updates
            key: Key to identify which cache to use
        """
        cache = self.caches[key]
        for i in range(self.n_t):
            j = cache['j'][i] * self.y_dim
            jbar = (cache['j'][i] ^ (1 << cache['r_min'][i])) * self.y_dim
            
            # Compute gradient contribution
            gi = 0.0
            for k in range(self.y_dim):
                gi += y_gradient[k] * (self.S[i][j + k] - self.S[i][jbar + k])
            
            # Compute delta and update input gradients
            delta = gi * up(cache['u_min'][i])
            x_gradient[self.anchors[i]['a'][cache['r_min'][i]]] += delta
            x_gradient[self.anchors[i]['b'][cache['r_min'][i]]] -= delta
            
            # Accumulate weight gradients instead of updating immediately
            for k in range(self.y_dim):
                weight_idx = j + k
                weight_key = (i, weight_idx)
                if weight_key not in self.v_weight_grads:
                    self.v_weight_grads[weight_key] = 0.0
                self.v_weight_grads[weight_key] += y_gradient[k]

    def apply_gradients(self, learning_rate: float) -> None:
        """
        Apply accumulated gradients to LUT weights.
        """
        for (i, weight_idx), grad in self.v_weight_grads.items():
            self.S[i][weight_idx] -= learning_rate * grad
        # Clear accumulated gradients after applying
        self.v_weight_grads.clear()


class PositionalEncoding:
    """
    Positional encoding with cache for attention mechanism.
    """
    
    def __init__(self, n_t: int = N_T, positional_dim: int = POSITIONAL_DIM, context_size: int = CONTEXT_SIZE):
        """
        Initialize positional encoding.
        
        Args:
            n_t: Number of detectors
            positional_dim: Dimension of positional encoding
            context_size: Maximum context size
        """
        self.n_t = n_t
        self.positional_dim = positional_dim
        self.context_size = context_size
        
        # Initialize positional encodings: [context_size - 1][n_t][positional_dim]
        self.encodings = []
        for pos in range(context_size - 1):
            pos_encoding = []
            for i in range(n_t):
                encoding = [random.uniform(-1.0, 1.0) for _ in range(positional_dim)]
                pos_encoding.append(encoding)
            self.encodings.append(pos_encoding)
        
        # Internal cache storage: dict[pos] -> cache dict
        self.caches = {}
    
    def cache_index(self, pos: int) -> None:
        """
        Cache lookup indices for positional encoding at position pos and store internally.
        
        Args:
            pos: Position index (0 to context_size - 2)
        """
        cache = {
            'j': [0] * self.n_t,
            'r_min': [0] * self.n_t,
            'u_min': [float('inf')] * self.n_t
        }
        
        u = self.encodings[pos]  # [n_t][positional_dim]
        
        for i in range(self.n_t):
            cache['j'][i] = 0
            cache['u_min'][i] = float('inf')
            
            for r in range(self.positional_dim):
                val = u[i][r]
                if val > 0:
                    cache['j'][i] |= (1 << r)
                if abs(val) < abs(cache['u_min'][i]):
                    cache['r_min'][i] = r
                    cache['u_min'][i] = val

        # print(f'cached positional info {cache} for position {pos}')

        self.caches[pos] = cache
    
    def update(self, pos: int, gradients: List[List[float]], learning_rate: float) -> None:
        """
        Update positional encoding weights.
        
        Args:
            pos: Position index
            gradients: Gradients for each detector and dimension [n_t][positional_dim]
            learning_rate: Learning rate
        """
        for i in range(self.n_t):
            for k in range(self.positional_dim):
                self.encodings[pos][i][k] -= learning_rate * gradients[i][k]


class AttentionHead:
    """
    Attention head using concatenated LUT (Q, K, PE).
    """
    
    def __init__(self, n_t: int = N_T, n_c: int = N_C, embedding_dim: int = EMBEDDING_DIM, 
                 positional_dim: int = POSITIONAL_DIM, context_size: int = CONTEXT_SIZE):
        """
        Initialize attention head.
        
        Args:
            n_t: Number of detectors
            n_c: Number of anchor pairs per detector
            embedding_dim: Embedding dimension
            positional_dim: Positional encoding dimension
            context_size: Context size
        """
        self.n_t = n_t
        self.n_c = n_c
        self.embedding_dim = embedding_dim
        self.positional_dim = positional_dim
        self.context_size = context_size
        
        # V LUT: concatenated [Q, K, PE] -> embedding_dim
        # Total input dimension: n_c (for Q) + n_c (for K) + positional_dim (for PE)
        total_n_c = n_c + n_c + positional_dim
        self.V = LUT(total_n_c, embedding_dim, n_t=n_t, n_c=n_c, embedding_dim=embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(n_t=n_t, positional_dim=positional_dim, context_size=context_size)
        # Accumulator for V weight gradients: dict[(table_index, weight_idx)] -> grad
        self.v_weight_grads = {}
        # Accumulator for positional encoding gradients:
        # Shape: [context_size - 1][n_t][positional_dim]
        self.pos_grad = [
            [[0.0] * self.positional_dim for _ in range(self.n_t)]
            for _ in range(self.context_size - 1)
        ]
    
    def _concatenate(self, q: int, k: int, pe: int) -> int:
        """Concatenate Q, K, PE indices into single lookup index."""
        return ((q << (self.n_c + self.positional_dim)) | (k << self.positional_dim) | pe) * self.V.y_dim
    
    def forward(self, x: List[List[float]], y: List[List[float]]) -> None:
        """
        Forward pass for attention.
        
        Args:
            x: Input embeddings [context_size][embedding_dim]
            y: Output embeddings [context_size][embedding_dim] (will be modified in-place)
        """
        # Cache indices for all positions
        for pos in range(self.context_size):
            self.V.cache_index(x[pos], key=pos)
            if pos < self.context_size - 1:
                self.positional_encoding.cache_index(pos)
        
        # Process all pairs (pos, pos1) where pos1 < pos
        for i in range(self.n_t):
            for pos1 in range(self.context_size):
                for pos in range(pos1 + 1, self.context_size):
                    cacheQ = self.V.caches[pos]
                    cacheK = self.V.caches[pos1]
                    cachePE = self.positional_encoding.caches[pos - pos1 - 1]

                    # Concatenated forward pass

                    j = self._concatenate(cacheQ['j'][i], cacheK['j'][i], cachePE['j'][i])
                    print(f'detector index {i}, pos (j) {pos}, pos1 (i) {pos1}, Q {cacheQ["j"][i]}, K {cacheK["j"][i]}, PE {cachePE["j"][i]}, concat_index {j // self.V.y_dim}')
                    for k in range(self.V.y_dim):
                        y[pos][k] += self.V.S[i][j + k]
    
    def backward(self, x_grad: List[List[float]], y_grad: List[List[float]], learning_rate: float) -> None:
        """
        Backward pass for attention.
        
        Args:
            x_grad: Input gradients [context_size][embedding_dim] (will be modified in-place)
            y_grad: Output gradients [context_size][embedding_dim]
            learning_rate: Learning rate
        """
        # Process all pairs (pos, pos1) where pos1 < pos
        for pos in range(1, self.context_size):
            for pos1 in range(pos):
                cacheQ = self.V.caches[pos]
                cacheK = self.V.caches[pos1]
                cachePE = self.positional_encoding.caches[pos - pos1 - 1]
                
                for i in range(self.n_t):
                    j_idx = self._concatenate(cacheQ['j'][i], cacheK['j'][i], cachePE['j'][i])
                    
                    # Determine which component to update (Q, K, or PE)
                    if abs(cacheQ['u_min'][i]) < abs(cacheK['u_min'][i]):
                        # Update Q
                        jbar = self._concatenate(
                            cacheQ['j'][i] ^ (1 << cacheQ['r_min'][i]),
                            cacheK['j'][i],
                            cachePE['j'][i]
                        )
                        
                        gi = 0.0
                        for k in range(self.V.y_dim):
                            gi += y_grad[pos][k] * (self.V.S[i][j_idx + k] - self.V.S[i][jbar + k])
                        
                        delta = gi * up(cacheQ['u_min'][i])
                        x_grad[pos][self.V.anchors[i]['a'][cacheQ['r_min'][i]]] += delta
                        x_grad[pos][self.V.anchors[i]['b'][cacheQ['r_min'][i]]] -= delta
                    else:
                        # Update K
                        jbar = self._concatenate(
                            cacheQ['j'][i],
                            cacheK['j'][i] ^ (1 << cacheK['r_min'][i]),
                            cachePE['j'][i]
                        )
                        
                        gi = 0.0
                        for k in range(self.V.y_dim):
                            gi += y_grad[pos][k] * (self.V.S[i][j_idx + k] - self.V.S[i][jbar + k])
                        
                        delta = gi * up(cacheK['u_min'][i])
                        x_grad[pos1][self.V.anchors[i]['a'][cacheK['r_min'][i]]] += delta
                        x_grad[pos1][self.V.anchors[i]['b'][cacheK['r_min'][i]]] -= delta
                    
                    # Update PE if it has smallest delta
                    if (
                        abs(cachePE['u_min'][i]) < abs(cacheQ['u_min'][i]) and
                        abs(cachePE['u_min'][i]) < abs(cacheK['u_min'][i])
                    ):
                        jbarPE = self._concatenate(
                            cacheQ['j'][i],
                            cacheK['j'][i],
                            cachePE['j'][i] ^ (1 << cachePE['r_min'][i])
                        )
                        
                        giPE = 0.0
                        for k in range(self.V.y_dim):
                            giPE += y_grad[pos][k] * (self.V.S[i][j_idx + k] - self.V.S[i][jbarPE + k])
                        
                        deltaPE = giPE * up(cachePE['u_min'][i])
                        self.pos_grad[pos - pos1 - 1][i][cachePE['r_min'][i]] += deltaPE
                    
                    # Accumulate V weight gradients instead of updating immediately
                    for k in range(self.V.y_dim):
                        weight_idx = j_idx + k
                        key = (i, weight_idx)
                        if key not in self.v_weight_grads:
                            self.v_weight_grads[key] = 0.0
                        self.v_weight_grads[key] += y_grad[pos][k]

    def apply_gradients(self, learning_rate: float) -> None:
        """
        Apply accumulated gradients to V LUT weights.
        """
        for (i, weight_idx), grad in self.v_weight_grads.items():
            self.V.S[i][weight_idx] -= learning_rate * grad
        # Update positional encodings using accumulated gradients
        for pos in range(self.context_size - 1):
            self.positional_encoding.update(pos, self.pos_grad[pos], learning_rate)

        # Clear accumulated gradients after applying
        self.v_weight_grads.clear()
        for rel_pos in range(self.context_size - 1):
            for det in range(self.n_t):
                for k in range(self.positional_dim):
                    self.pos_grad[rel_pos][det][k] = 0.0


class TokenEmbedder:
    """
    Token embedding layer.
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM):
        """
        Initialize token embedder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings: [vocab_size][embedding_dim]
        self.embeddings = []
        for i in range(vocab_size):
            embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]
            self.embeddings.append(embedding)
    
    def embed(self, token: int) -> List[float]:
        """
        Embed a token.
        
        Args:
            token: Token index
        
        Returns:
            Embedding vector
        """
        return self.embeddings[token]


class _GTLUTTransformer:
    """
    Complete transformer model using LUT layers.
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM,
                 context_size: int = CONTEXT_SIZE, positional_dim: int = POSITIONAL_DIM,
                 num_layers: int = NUM_LAYERS, num_heads: int = NUM_HEADS,
                 n_t: int = N_T, n_t_a: int = N_T, n_c: int = N_C, batch_size: int = 1):
        """
        Initialize model.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            context_size: Context size
            positional_dim: Positional encoding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            n_t: Number of detectors
            n_c: Number of anchor pairs per detector
            batch_size: Batch size for processing
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.positional_dim = positional_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_t = n_t
        self.n_t_a = n_t_a
        self.n_c = n_c
        self.batch_size = batch_size
        
        # Token embedder
        self.token_embedder = TokenEmbedder(vocab_size, embedding_dim)
        
        # Transformer layers
        self.layers = []
        for l in range(num_layers):
            layer = {
                'ffn': LUT(n_c, embedding_dim, n_t=n_t, n_c=n_c, embedding_dim=embedding_dim),
                'heads': []
            }
            for h in range(num_heads):
                head = AttentionHead(
                    n_t=n_t_a, n_c=n_c, embedding_dim=embedding_dim,
                    positional_dim=positional_dim, context_size=context_size
                )
                layer['heads'].append(head)
            self.layers.append(layer)
        
        # Unembedder
        self.unembedder = LUT(n_c, vocab_size, n_t=n_t, n_c=n_c, embedding_dim=embedding_dim)
        
        # State variables
        # tokens: [batch_size][context_size + 1]
        # z:      [batch_size][context_size][embedding_dim]
        # output: [batch_size][context_size][vocab_size]
        self.tokens = [[0] * (context_size + 1) for _ in range(self.batch_size)]
        self.z = [
            [[0.0] * embedding_dim for _ in range(context_size)]
            for _ in range(self.batch_size)
        ]
        self.output = [
            [[0.0] * vocab_size for _ in range(context_size)]
            for _ in range(self.batch_size)
        ]
    
    def forward(self, bs=None) -> None:
        """
        Forward pass.
        """

        if bs is None:
            bs = self.batch_size

        for b in range(bs):
            # Process through transformer layers
            for l in range(self.num_layers):
                # Attention: resnet connection (add to z)
                x = [row.copy() for row in self.z[b]]  # Copy for attention input

                for h in range(self.num_heads):
                    y = [[0.0] * self.embedding_dim for _ in range(self.context_size)]
                    self.layers[l]['heads'][h].forward(x, y)
                    print(f'layer {l}, head {h}, gt attention output: {y}')
                    # Resnet connection: add to z
                    for pos in range(self.context_size):
                        for k in range(self.embedding_dim):
                            self.z[b][pos][k] += y[pos][k]

                # FFN: resnet connection (add to z)
                for pos in range(self.context_size):
                    self.layers[l]['ffn'].cache_index(self.z[b][pos], key=pos)
                    self.layers[l]['ffn'].forward(self.z[b][pos], key=pos)

            # Unembedder
            for pos in range(self.context_size):
                self.output[b][pos] = [0.0] * self.vocab_size
                self.unembedder.cache_index(self.z[b][pos], key=pos)
                self.unembedder.forward(self.output[b][pos], key=pos)
        
    def backward(self, learning_rate: float) -> None:
        """
        Backward pass.
        
        Args:
            learning_rate: Learning rate for weight updates
        """
        # x_grad: [batch_size][context_size][embedding_dim]
        x_grad = [
            [[0.0] * self.embedding_dim for _ in range(self.context_size)]
            for _ in range(self.batch_size)
        ]

        # Process all batches - accumulate gradients
        for b in range(self.batch_size):
            # Unembedder backward
            for pos in range(self.context_size):
                self.unembedder.backward(
                    x_grad[b][pos], self.output[b][pos], learning_rate, key=pos
                )

            # Process layers in reverse order
            for l in range(self.num_layers - 1, -1, -1):
                # FFN backward
                y_grad = [row.copy() for row in x_grad[b]]  # Copy for resnet connection
                for pos in range(self.context_size):
                    self.layers[l]["ffn"].backward(
                        x_grad[b][pos], y_grad[pos], learning_rate, key=pos
                    )

                # Attention backward
                y_grad = [row.copy() for row in x_grad[b]]  # Copy for resnet connection
                for h in range(self.num_heads):
                    self.layers[l]["heads"][h].backward(
                        x_grad[b], y_grad, learning_rate
                    )
        
        # Apply accumulated gradients after processing all batches
        # Unembedder
        self.unembedder.apply_gradients(learning_rate)
        
        # Process layers in reverse order
        for l in range(self.num_layers - 1, -1, -1):
            # FFN
            self.layers[l]["ffn"].apply_gradients(learning_rate)
            
            # Attention heads
            for h in range(self.num_heads):
                self.layers[l]["heads"][h].apply_gradients(learning_rate)
    
    def training_step(self, learning_rate: float) -> None:
        """
        Single training step: forward + backward.
        
        Args:
            learning_rate: Learning rate
        """
        self.forward()
        
        # Compute softmax and convert output to gradients for all batches
        for b in range(self.batch_size):
            for pos in range(self.context_size):
                self.output[b][pos] = softmax(self.output[b][pos], temperature=1.0)
                self.output[b][pos][self.tokens[b][pos + 1]] -= 1.0  # Output becomes a gradient
        
        self.backward(learning_rate)
    
    def load_snippet(self, text_or_batch, start_idx_or_batch_idx: int = None) -> None:
        """
        Load snippet(s) of text into the model.
        
        Args:
            text_or_batch: Either:
                - Text data as list of token indices (single snippet, requires start_idx_or_batch_idx)
                - List of token lists (batch of snippets, one per batch item)
            start_idx_or_batch_idx: If text_or_batch is a single list, this is the starting index.
                                   If text_or_batch is a batch, this is ignored.
        """
        # Check if it's a batch (list of lists) or a single snippet (list of ints)
        if isinstance(text_or_batch[0], list):
            # Batch mode: text_or_batch is a list of token lists
            batch_tokens = text_or_batch
            if len(batch_tokens) != self.batch_size:
                raise ValueError(f"Batch size mismatch: expected {self.batch_size}, got {len(batch_tokens)}")
            
            for b in range(self.batch_size):
                tokens = batch_tokens[b]
                if len(tokens) < self.context_size + 1:
                    raise ValueError(f"Snippet too short: need {self.context_size + 1} tokens, got {len(tokens)}")
                
                for pos in range(self.context_size):
                    self.z[b][pos] = self.token_embedder.embed(tokens[pos])
                    self.tokens[b][pos] = tokens[pos]
                self.tokens[b][self.context_size] = tokens[self.context_size]
        else:
            # Single snippet mode: text_or_batch is a list of token indices
            if start_idx_or_batch_idx is None:
                raise ValueError("start_idx_or_batch_idx required when loading single snippet")
            
            text = text_or_batch
            start_idx = start_idx_or_batch_idx
            b = 0  # For backward compatibility, load into batch 0

            for pos in range(self.context_size):
                self.z[b][pos] = self.token_embedder.embed(text[start_idx + pos])
                self.tokens[b][pos] = text[start_idx + pos]
            self.tokens[b][self.context_size] = text[start_idx + self.context_size]
    
    def inference(self) -> int:
        """
        Run inference and sample next token.
        
        Returns:
            Sampled token index
        """
        # NOTE: for now we only use batch index 0.
        b = 0

        # Reset output
        for pos in range(self.context_size):
            self.output[b][pos] = [0.0] * self.vocab_size

        self.forward(bs=1)
        
        # Softmax with temperature 0.4
        probs = softmax(self.output[b][self.context_size - 1], temperature=0.4)
        return sample(probs)
    
    def prompt_response(self, prompt: List[int], prompt_length: int) -> List[int]:
        """
        Generate response from prompt.
        
        Args:
            prompt: Prompt tokens
            prompt_length: Length of response to generate
        
        Returns:
            Generated tokens
        """
        # NOTE: for now we only use batch index 0.
        b = 0

        prompt_copy = prompt[:self.context_size].copy()
        if len(prompt_copy) < self.context_size:
            prompt_copy.extend([0] * (self.context_size - len(prompt_copy)))
        
        generated = []
        
        for i in range(prompt_length):
            # Load prompt into model
            for pos in range(self.context_size):
                self.z[b][pos] = self.token_embedder.embed(prompt_copy[pos])
            
            # Generate next token
            response = self.inference()
            generated.append(response)
            
            # Shift prompt and add response
            prompt_copy = prompt_copy[1:] + [response]
        
        return generated


class TrainingData:
    """
    Training data loader matching the C code's TrainingData structure.
    """
    
    def __init__(self, filepath: str, context_size: int = CONTEXT_SIZE, testing_length: int = TESTING_LENGTH):
        """
        Load training data from file.
        
        Args:
            filepath: Path to text file
            context_size: Context size
            testing_length: Number of testing samples
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error opening training datafile {filepath}")
        
        print(f"Successfully opened training data file {filepath}")
        
        # Convert bytes to token indices (0-255)
        self.data = list(data)
        self.length = len(self.data) - (context_size + 1)
        
        # Reserve some data for testing
        self.reserved_for_testing = [False] * self.length
        self.testing_input_data = []
        
        # Randomly select testing positions
        random.seed(42)
        for i in range(testing_length):
            idx = random.randint(0, self.length - 1)
            self.testing_input_data.append(idx)
            # Mark reserved positions (with context around them)
            start = max(0, idx - context_size)
            end = min(self.length, idx + context_size + 1)
            for j in range(start, end):
                self.reserved_for_testing[j] = True
        
        print(f"Successfully loaded training data")
        print(f"Total tokens: {len(self.data)}")
        print(f"Training length: {self.length}")
        print(f"Reserved for testing: {sum(self.reserved_for_testing)}")
    
    def get_random_training_index(self) -> int:
        """Get a random training index that's not reserved for testing."""
        while True:
            idx = random.randint(0, self.length - 1)
            if not self.reserved_for_testing[idx]:
                return idx
    
    def get_random_training_indices(self, batch_size: int) -> List[int]:
        """Get a batch of random training indices that are not reserved for testing."""
        indices = []
        while len(indices) < batch_size:
            idx = random.randint(0, self.length - 1)
            if not self.reserved_for_testing[idx]:
                indices.append(idx)
        return indices


def train_model(model: _GTLUTTransformer, training_data: TrainingData, num_steps: int = 10000,
                log_interval: int = 1000, loss_file: str = "loss.csv", batch_size: int = 1) -> None:
    """
    Training loop matching the C code's main function.
    
    Args:
        model: _GTLUTTransformer to train
        training_data: TrainingData instance
        num_steps: Number of training steps
        log_interval: Interval for logging and validation
        loss_file: File to save loss values
        batch_size: Number of samples per training step
    """
    # Validate batch size matches model
    if model.batch_size != batch_size:
        raise ValueError(f"Model batch_size ({model.batch_size}) must match training batch_size ({batch_size})")
    
    # Initialize loss file
    with open(loss_file, 'w') as f:
        f.write("step,loss\n")
    
    print(f"Starting training for {num_steps} steps with batch_size={batch_size}...")
    
    for step in range(num_steps):
        # Get batch of random training indices
        start_indices = training_data.get_random_training_indices(batch_size)
        
        # Load batch of snippets into model
        batch_tokens = []
        for start_idx in start_indices:
            snippet = training_data.data[start_idx:start_idx + model.context_size + 1]
            batch_tokens.append(snippet)
        model.load_snippet(batch_tokens)
        
        # Get learning rate from scheduler
        lr = learning_rate_scheduler(step)
        
        # Training step
        model.training_step(lr)
        
        # Validation and logging
        if (step + 1) % log_interval == 0:
            print(f"...validating... ", end='', flush=True)
            loss_average = 0.0
            
            # Compute validation loss (using batch_size=1 for validation)
            for i, test_idx in enumerate(training_data.testing_input_data):
                model.load_snippet(training_data.data, test_idx)
                model.forward(bs=1)
                
                # Compute loss: negative log probability of correct token
                probs = softmax(model.output[0][model.context_size - 1], temperature=1.0)
                correct_token = training_data.data[test_idx + model.context_size]
                loss_average += -math.log(max(probs[correct_token], 1e-10))
            
            loss_average /= len(training_data.testing_input_data)
            
            # Save loss to file
            with open(loss_file, 'a') as f:
                f.write(f"{step},{loss_average:.6f}\n")
            
            print(f"\rt={step//1000},000, loss={loss_average:.3f}: ", end='', flush=True)
            
            # Generate sample text
            prompt_text = "Testing prompt: Once upon a time "
            prompt_tokens = [ord(c) for c in prompt_text[:model.context_size]]
            if len(prompt_tokens) < model.context_size:
                prompt_tokens.extend([0] * (model.context_size - len(prompt_tokens)))
            
            generated = model.prompt_response(prompt_tokens, prompt_length=80)
            generated_text = ''.join([chr(t) if 32 <= t < 127 else '?' for t in generated])
            print(f"{prompt_text}{generated_text}")
        
        if (step + 1) % 100 == 0:
            print(f"\rt={step}", end='', flush=True)
    
    print(f"\nTraining completed!")


def main():
    """Main function matching the C code's main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train text LUT transformer model')
    parser.add_argument(
        '--data', type=str, default='tinyshakespeare.txt',
        help='Path to training data file'
    )
    parser.add_argument(
        '--steps', type=int, default=10000,
        help='Number of training steps'
    )
    parser.add_argument(
        '--log-interval', type=int, default=1000,
        help='Interval for logging and validation'
    )
    parser.add_argument(
        '--loss-file', type=str, default='loss.csv',
        help='File to save loss values'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--single-head', dest='multi_head', action='store_false', default=True,
        help='Use single-head mode (default: multi-head mode)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass
    
    # Load training data
    try:
        training_data = TrainingData(args.data, context_size=CONTEXT_SIZE, testing_length=TESTING_LENGTH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create model
    if args.multi_head:
        model = _GTLUTTransformer(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            context_size=CONTEXT_SIZE,
            positional_dim=POSITIONAL_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            n_t=N_T,
            n_t_a=N_T,
            n_c=N_C,
            batch_size=args.batch_size
        )
    else:
        model = _GTLUTTransformer(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            context_size=CONTEXT_SIZE,
            positional_dim=POSITIONAL_DIM,
            num_layers=NUM_LAYERS,
            num_heads=1,
            n_t=N_T,
            n_t_a=N_T * NUM_HEADS,
            n_c=N_C,
            batch_size=args.batch_size
        )
    
    print(f"Model created:")
    print(f"  Context size: {CONTEXT_SIZE}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Num heads: {NUM_HEADS if args.multi_head else 1}")
    print(f"  N_T: {N_T}")
    print(f"  N_T_A: {N_T if args.multi_head else N_T * NUM_HEADS}")
    print(f"  N_C: {N_C}")
    print(f"  Batch size: {args.batch_size}")
    
    # Train model
    train_model(
        model, training_data, num_steps=args.steps,
        log_interval=args.log_interval, loss_file=args.loss_file,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
