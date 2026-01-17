# Transformer From Scratch – GPT-style Language Model

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Data Processing and Dataset](#3-data-processing-and-dataset)
4. [Training Setup](#4-training-setup)
5. [Progressive Model Construction](#5-progressive-model-construction)
    1. [Bigram Model (Baseline)](#51-bigram-model-baseline)
    2. [Adding Positional Embeddings](#52-adding-positional-embeddings)
    3. [Single-Head Attention](#53-single-head-attention)
    4. [Multi-Head Attention](#54-multi-head-attention)
    5. [Feed-Forward Network](#55-feed-forward-network)
    6. [Multiple Transformer Blocks](#56-multiple-transformer-blocks)
    7. [Residual Connections and Layer Normalization](#57-residual-connections-and-layer-normalization)
    8. [Dropout](#58-dropout)
6. [Final Model Results](#6-final-model-results)
7. [Future Improvements](#7-future-improvements)
8. [Conclusion](#8-conclusion)


## 1. Project Overview

This project is a fast, educational implementation of a GPT-style language model built **from scratch in PyTorch**.  
The main goal was not to achieve state-of-the-art performance, but to **understand and analyze how individual Transformer components affect performance**.

The model is trained in a **character-level language modeling** setup, where it learns to predict the next character given a fixed-length context window. Starting from a simple bigram baseline, the architecture is gradually extended with core Transformer elements such as positional embeddings, self-attention, feed-forward layers, residual connections, layer normalization, and dropout.

Each architectural change is evaluated independently, and training/validation losses are recorded to observe how model capacity, stability, and generalization evolve. The final result is a compact GPT-like model trained on a ~1M character dataset, together with an analysis of overfitting behavior.

---

## 2. Model Architecture

The architecture of this model is based on the Transformer design introduced in the paper  
**“Attention Is All You Need”** (Vaswani et al., 2017).

While the original publication presents both encoder and decoder stacks, this project implements a **decoder-only Transformer**, following the design used in GPT-style language models.

At a high level, the data flow through the model can be summarized as:

```
Input tokens
   ↓
Token + Positional Embedding
   ↓
[N × Transformer Block]
   ↓
LayerNorm
   ↓
Linear head → logits
```

Each **Transformer Block** consists of two main components:
- **Multi-Head Self-Attention**, which allows the model to attend to different positions within the input sequence and capture long-range dependencies,
- **Feed-Forward Network**, which applies a position-wise non-linear transformation to the representations produced by the attention mechanism.

In this implementation, **pre-layer normalization** is used. Layer normalization is applied *before* both the self-attention module and the feed-forward network, following the design commonly used in modern GPT-style models. This variant improves training stability and gradient flow, especially for deeper architectures.

The model operates in an **autoregressive** setting. During both training and inference, each token is only allowed to attend to tokens at earlier positions in the sequence. This causal constraint is enforced using a lower-triangular attention mask.

---

## 3. Data Processing and Dataset

The model is trained using a **character-level language modeling** approach. Each character from the input text is treated as an individual token, and the task is to predict the next character given a fixed-length context window.

### Tokenization

The dataset vocabulary is constructed from the set of all unique characters appearing in the input text.  
Each character is mapped to a unique integer identifier, forming a simple lookup-based tokenizer. During decoding, the process is reversed to reconstruct text from model predictions.

### Sequence Construction

Training samples are generated using a **sliding window** over the tokenized text. For a given position in the dataset:
- the input sequence consists of `block_size` consecutive tokens,
- the target sequence is the same sequence shifted by one position.

This setup allows the model to learn next-token prediction in an autoregressive manner.

### Train / Validation Split

The dataset is split into training and validation subsets using a fixed ratio:
- 90% of the text is used for training,
- 10% is held out for validation.

The split is performed at the raw text level to preserve the natural sequential structure of the data.

### Iteration-Based Training

Instead of training in full epochs, the model is trained using a fixed number of **training iterations (steps)**.  
At each iteration, a mini-batch of subsequences is sampled from the dataset.

Due to the relatively small batch size, the number of batches per iteration would be very large, and training the model over full epochs would take a long time. Therefore, training is conducted using a fixed number of iterations to allow more control over training time and better evaluate performance at regular intervals.

---

## 4. Training Setup

The training process is designed around a **fixed number of iterations (steps)** rather than full epochs, which is particularly useful when working with large datasets and limited hardware. In each iteration, a mini-batch is sampled from the dataset, and the model is updated using backpropagation.

### Loss Function

The model is trained using **cross-entropy loss**, which is the standard loss function for classification tasks, including language modeling. This loss function compares the predicted logits (raw output scores) for each token with the actual target token, computing the error that will be backpropagated through the network.

### Optimizer

The **AdamW** optimizer is used for training. AdamW is a variant of the Adam optimizer that decouples weight decay from the optimization steps, making it more effective for training large models like Transformers. It is particularly well-suited for training deep models with sparse gradients, which is common in natural language processing tasks.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

### Evaluation Strategy

To monitor the model's performance during training, we evaluate the model on the validation set at regular intervals:
- Every EVAL_INTERVAL steps, the model is evaluated on both the train and validation datasets.
- This allows tracking the loss values on both datasets, and helps identify overfitting or underfitting at an early stage.

#### DataLoaders
```python
# Read text
with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))

# Split data into train and valid set
n = int(SPLIT_RATIO*len(text))
train_text = text[:n]
valid_text = text[n:]

# Initialize tokenizer
tokenizer = CharTokenizer(chars)

# Create datasets
train_dataset = TextLoader(train_text, tokenizer=tokenizer, block_size=BLOCK_SIZE, device=DEVICE)
valid_dataset = TextLoader(valid_text, tokenizer=tokenizer, block_size=BLOCK_SIZE, device=DEVICE)

# Initialize dataloaders
train_dataloader = iter(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))
valid_dataloader = iter(DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True))
```

#### Estiamte Training and Validation Loss
```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for dataset in ['train', 'valid']:
        losses = torch.zeros(EVAL_ITERS)
        dataloader = train_dataloader if dataset == 'train' else valid_dataloader
        for k in range(EVAL_ITERS):
            X, Y = next(dataloader)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[dataset] = losses.mean()
    model.train()
    return out
```
The complete code for training the model can be found in the `train.py` file.

---

The hyperparameters of the model that were used to check how the model performs after adding individual sections are listed below.
```python
BLOCK_SIZE = 8
BATCH_SIZE = 4
N_TRANSFORMER_BLOCKS = 4
N_EMBED = 32
NUM_HEADS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
N_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
```
Individual parameters were added along with the added sections of the model. 

## 5. Progressive Model Construction

The model was developed incrementally, with each new architectural component added one step at a time. This section describes the key changes made to the model and the impact on its performance.

### 5.1 Bigram Model (Baseline)

The simplest model in this project was a **bigram model**, where each token is predicted based solely on the previous token. This served as a baseline for training and provided a reference point to evaluate improvements made with each architectural change.

The **bigram model** only uses **token embeddings**, and the prediction is based on a simple lookup in the embedding table for each token in the sequence. It doesn't use attention or any advanced features, making it computationally cheap and easy to train. 

Below is the code for the bigram model, which only includes token embeddings and a basic forward pass:

```python
class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        logits = self.token_embedding_table(x)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # (batch*block_size, vocab_size)
            targets = targets.view(B*T)   # (batch*block_size, 1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx (B, T)
        for _ in range(max_new_tokens):
            # crop idx to the newest block_size tokens
            idx_cropped = idx[:, -self.position_embedding_table.num_embeddings:]
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append predicted token to the running sequence
            idx = torch.cat((idx, next_idx), dim=-1)
        return idx
```

- **Training Loss**: 2.67
- **Validation Loss**: 2.68

### 5.2 Adding Positional Embeddings

The next major improvement was the addition of **positional embeddings**. In Transformer models, since the self-attention mechanism does not inherently capture the order of the tokens in the sequence, we need to explicitly provide positional information to the model. This is achieved by adding **positional embeddings** to the token embeddings, allowing the model to differentiate between tokens based on their positions in the sequence.

The positional embeddings are added to the token embeddings before being fed into the model, enabling it to consider both the content of the tokens and their relative positions.

```python
class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_embed: int
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
```

- **Training Loss**: 2.5
- **Validation Loss**: 2.52
  
The positional embeddings allow the model to distinguish between tokens based on their position in the sequence. 
By adding these embeddings to the input token embeddings, the model can now better understand the sequence structure, 
enabling it to make predictions that take into account the order of the tokens. This improvement is fundamental for models like Transformers, 
which do not have an inherent notion of sequence order.

### 5.3 Single-Head Attention

After introducing positional embeddings, the next key addition was **single-head attention**. The attention mechanism is at the heart of the Transformer architecture and is crucial for capturing relationships between different positions in the input sequence.

In **self-attention**, each token in the sequence can attend to all other tokens to form a richer representation. For each token, we compute three vectors:
- **Query (Q)**: Represents the current token for which we are computing the attention.
- **Key (K)**: Represents all tokens in the sequence that the query might attend to.
- **Value (V)**: Represents the actual information (data) we aggregate when computing the attention.

#### How Attention Works:

Let’s consider an example sentence: **"The dog is running in the park."** We will focus on the token "dog" and explain how the self-attention mechanism works.

1. **Query (Q)** for the token "dog" is the vector representation of "dog". This represents the current token for which we are computing attention.

2. **Key (K)** represents all other tokens in the sequence, including "the", "is", "running", "in", and "the park". These tokens are the ones the current token ("dog") might have the relation with other tokens..

3. **Value (V)** represents the actual data associated with each token, which will be aggregated to form the updated representation for the token "dog".

#### The Self-Attention Calculation:

1. **Dot Product**: The **query (Q)** for the token "dog" is compared with the **keys (K)** for all other tokens in the sequence, using a **dot product**. This gives us a score for how much each token in the sequence should be attended to by "dog".

2. **Scaling**: The dot product scores are scaled by the square root of the dimension of the key vectors (`head_size`), ensuring that the values are in a reasonable range.

3. **Softmax**: The scaled dot product scores are passed through the **softmax function** to normalize them into probabilities. These probabilities indicate how much attention the token "dog" should pay to other tokens in the sequence (like "is", "running", or "in the park").

4. **Weighted Sum**: The attention probabilities are used to weight the **value vectors (V)**, and a weighted sum is computed for each token. This sum forms the updated representation of the token "dog", which now includes information from other tokens in the sequence (like "is", "running", "in the park").

#### Single-Head Attention Code

In a single attention head, all the queries, keys, and values are processed simultaneously, and the output is a single weighted sum of the values based on the attention scores. This process is repeated for every token in the sequence.

Below is the implementation for the **Single-Head Attention** mechanism:

```python
class Head(nn.Module):
    def __init__(self, block_size: int, n_embed: int, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Lower triangular mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # B = batch size, T = sequence length, C = embedding size

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores (dot product)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)

        # Apply the lower triangular mask to ensure causality (no future tokens can be attended to)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # Apply softmax to get attention probabilities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        v = self.value(x)  # (B, T, head_size)

        # Compute the output as the weighted sum of value vectors
        out = wei @ v  # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_embed: int,
        head_size: int
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa = Head(block_size, n_embed, head_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa(x)  # (B, T, C)
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
```

In a single-head attention setup, all queries, keys, and values share the same attention weights. This makes it easier to compute, but limits the model’s ability to capture diverse relationships within the input. 
Each token only attends to the sequence through a single perspective, which means the model might miss more complex relationships in the data.  

Even with these limitations, single-head attention still improves the model over simpler approaches (such as bigram models) and provides valuable insights into how self-attention works.
- **Training Loss**: 2.47
- **Validation Loss**: 2.45

### 5.4 Multi-Head Attention

After adding single-head attention, the next improvement was **multi-head attention**. This is a core component of the Transformer architecture, where the model uses multiple attention heads to simultaneously focus on different parts of the sequence. Each attention head can attend to different aspects of the sequence, which allows the model to capture more complex relationships and dependencies.

In **multi-head attention**, multiple **single-head attention** mechanisms are used in parallel, each with its own **query**, **key**, and **value** projections. The outputs of all attention heads are concatenated and passed through a final linear projection to produce the final result.

#### Head Size Calculation

The **head size** for each attention head is calculated by dividing the embedding dimension (`n_embed`) by the number of heads (`num_heads`):
- **head_size = n_embed // num_heads**

This ensures that the total dimension after concatenating the outputs of all heads is equal to the original embedding dimension (`n_embed`).

For example, if `n_embed = 128` and `num_heads = 8`, then the **head size** for each attention head will be `128 // 8 = 16`. This allows the model to maintain the same final embedding size while using multiple heads.

**Query (Q), Key (K), Value (V) Vectors**: For each token, multiple queries, keys, and values are computed in parallel, each using a separate attention head. This allows the model to attend to different parts of the sequence at the same time.
**Concatenation and Linear Projection**: The outputs of all attention heads are concatenated and passed through a final linear projection. This allows the model to aggregate information from different parts of the sequence, improving its ability to capture complex relationships.

#### Multi-Head Attention Code

Below is the implementation of **multi-head attention** where the attention heads are processed in parallel, and their results are concatenated and projected to produce the final output.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int, num_heads: int, n_embed: int):
        super().__init__()
        head_size = n_embed // num_heads  # Calculate head_size to ensure the final dimension after concatenation is correct
        self.heads = nn.ModuleList([Head(block_size, n_embed, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)  # Final linear projection to combine all heads

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate the outputs from all heads (B, T, num_heads*head_size)
        out = self.proj(out)  # (B, T, C)
        return out

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_embed: int,
        head_size: int
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa = MultiHeadAttention(block_size, num_heads, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

The rest of the code remains unchanged.
```

The main advantage of multi-head attention is that it allows the model to attend to different parts of the sequence simultaneously, improving its ability to capture various relationships. 
Each attention head can focus on a different part of the sequence, which helps the model understand the context better.

- **Training Loss**: 2.36
- **Validation Loss**: 2.38

### 5.5 Feed-Forward Network

After adding attention mechanisms, the next improvement to the model was the addition of a **feed-forward network** (FFN) that follows the attention layer. The purpose of the FFN is to process each token's representation independently and apply non-linear transformations to capture more complex patterns.

In the Transformer architecture, the FFN is applied **position-wise**, meaning that the same network is applied to each token in the sequence independently. This helps the model learn more abstract representations of the input data.

#### Feed-Forward Network Structure

The FFN consists of two fully connected layers with a ReLU activation in between:
1. **First Layer**: A linear transformation that increases the dimensionality by a factor of 4 (i.e., `4 * n_embed`).
2. **ReLU Activation**: A non-linearity applied after the first linear transformation.
3. **Second Layer**: A linear transformation that projects the output back to the original embedding size (`n_embed`).

This structure helps the model learn complex transformations while maintaining the same dimensionality for each token’s representation.

#### Feed-Forward Network Code

Below is the implementation for the **feed-forward network**:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Increase dimensionality by a factor of 4
            nn.ReLU(),  # Non-linearity
            nn.Linear(4 * n_embed, n_embed),  # Project back to original size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffwd(x)  # (B, T, C)

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_embed: int,
        head_size: int
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa = MultiHeadAttention(block_size, num_heads, n_embed)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa(x)  # (B, T, C)
        x = self.ffwd(x)  (B, T, C)
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

### 5.6 Multiple Transformer Blocks

After adding a feed-forward network, the next step was to stack **multiple Transformer blocks**. In the original Transformer architecture, the model is composed of a stack of identical layers, where each layer consists of a **self-attention** mechanism followed by a **feed-forward network**. This structure allows the model to process and learn increasingly complex representations of the input sequence.

The number of Transformer blocks (also known as **layers**) determines the model's capacity. By adding more blocks, we give the model more layers through which it can learn higher-level features of the data. 

#### Why Multiple Blocks?

- **Learning Hierarchical Features**: Each Transformer block learns increasingly abstract representations of the input sequence. The more blocks you stack, the better the model can capture complex patterns and dependencies in the data.
- **Improved Expressiveness**: With more blocks, the model has more capacity to transform and refine the token representations, leading to a richer understanding of the sequence.
- **Gradual Complexity**: The stacked blocks enable the model to learn complex relationships step-by-step. Each block can refine the information passed from the previous one, resulting in more sophisticated representations after the final block.

#### Stacking Multiple Transformer Blocks

To increase the model's depth, multiple Transformer blocks are stacked in sequence. Each block processes the output of the previous one, learning increasingly complex representations of the input sequence.

#### Transformer Block Code

Each **Transformer block** consists of a **self-attention layer** (as implemented in the `MultiHeadAttention` class) and a **feed-forward network** (as implemented in the `FeedForward` class). Below is the implementation for a single Transformer block, which is repeated in the model architecture:

```python
class TransformerBlock(nn.Module):
    def __init__(self, block_size: int, num_heads: int, n_embed: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(block_size, num_heads, n_embed, dropout)  # Self-attention layer
        self.ffwd = FeedForward(n_embed, dropout)  # Feed-forward network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sa(x)    # Residual connection with self-attention
        x = self.ffwd(x)  # Residual connection with feed-forward network
        return x

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_transformer_blocks: int, 
        n_embed: int, 
        num_heads: int
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(block_size, num_heads, n_embed, dropout) for _ in range(n_transformer_blocks)])
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

```

- **Training Loss**: 2.53
- **Validation Loss**: 2.54

### 5.7 Residual Connections and Layer Normalization

After adding multiple Transformer blocks, the next key improvement was the introduction of **residual connections** and **layer normalization**. These techniques are essential for training deeper models like Transformers, as they help stabilize training and improve gradient flow.

#### Residual Connections

In deep neural networks, especially in architectures like Transformers, **residual connections** allow the model to bypass certain layers, facilitating the flow of gradients during backpropagation. This prevents the vanishing gradient problem and helps the model learn more effectively, even when the architecture is very deep.

- **How They Work**: A residual connection adds the input of a layer directly to its output. This allows the model to learn an identity mapping if needed, which makes it easier to train deeper networks.
  
In the Transformer, residual connections are applied to both the **self-attention** layer and the **feed-forward network**, making it easier for gradients to propagate back through the network.

#### Layer Normalization

**Layer normalization** is applied after each residual connection to stabilize the learning process and improve training efficiency. It normalizes the activations within each layer, ensuring that the input to each layer has a consistent distribution, which speeds up convergence and improves generalization.

- **How It Works**: Layer normalization normalizes the activations across each feature dimension, rather than across the batch dimension as in batch normalization. This makes it more suitable for sequence data, where the batch size may vary or where sequence length can differ.

Together, **residual connections** and **layer normalization** improve the flow of gradients and ensure that the model can train more efficiently, even with many layers.

#### Code Implementation

Below is the code that implements **residual connections** and **layer normalization** in the Transformer model:

```python
class TransformerBlock(nn.Module):
    def __init__(self, block_size: int, num_heads: int, n_embed: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(block_size, num_heads, n_embed, dropout)  # Self-attention layer
        self.ffwd = FeedForward(n_embed, dropout)  # Feed-forward network
        self.ln1 = nn.LayerNorm(n_embed)  # Layer normalization before self-attention
        self.ln2 = nn.LayerNorm(n_embed)  # Layer normalization before feed-forward network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply residual connection after self-attention and layer normalization
        x = x + self.sa(self.ln1(x))    # Residual connection with self-attention
        # Apply residual connection after feed-forward network and layer normalization
        x = x + self.ffwd(self.ln2(x))  # Residual connection with feed-forward network
        return x

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_transformer_blocks: int, 
        n_embed: int, 
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(block_size, num_heads, n_embed, dropout) for _ in range(n_transformer_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # (batch*block_size, vocab_size)
            targets = targets.view(B*T)   # (batch*block_size, 1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

- **Training Loss**: 2.24
- **Validation Loss**: 2.29

### 5.8 Dropout

The final improvement made to the model was the addition of **dropout**, a regularization technique used to prevent overfitting. Dropout helps the model generalize better to unseen data by randomly setting a fraction of the input units to zero during training, forcing the network to rely on different features and not overfit to specific parts of the data.

#### How Dropout Works

1. **Randomly Drop Neurons**: During each forward pass, a fraction of the neurons in the network are randomly "dropped," meaning their output is set to zero. This prevents the model from becoming overly reliant on specific neurons.
   
2. **Dropout Rate**: The dropout rate is the probability of a neuron being dropped. It is typically set as a hyperparameter (e.g., 0.2 means that 20% of the neurons are randomly dropped during training).

3. **Training vs. Inference**: During **training**, dropout is active, and neurons are randomly dropped. However, during **inference**, dropout is turned off, and the full network is used to make predictions.

#### Why Dropout?

- **Preventing Overfitting**: By randomly dropping neurons, dropout prevents the model from becoming too specialized to the training data. This encourages the model to learn more robust and generalizable features.
- **Improved Generalization**: Dropout helps the model learn more diverse features and prevents it from memorizing the training data, improving performance on unseen data.

#### Dropout in the Code

Below is the implementation of **dropout** in the particular layers of the Transformer model:

```python
class Head(nn.Module):
    def __init__(self, block_size: int, n_embed: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # (B, T, T)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int, num_heads: int, n_embed: int, dropout: float):
        super().__init__()
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList([Head(block_size, n_embed, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads*head_size) ---> (B, T, C)
        out = self.proj(out)  # (B, T, C)
        out = self.dropout(out)  # (B, T, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Increase dimensionality by a factor of 4
            nn.ReLU(),  # Non-linearity
            nn.Linear(4 * n_embed, n_embed),  # Project back to original size
            nn.Dropout(dropout)  # Dropout to prevent overfitting
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffwd(x)  # (B, T, C)
```

- **Training Loss**: 2.40
- **Validation Loss**: 2.42

### Conclusion

In this section, we’ve walked through the progressive improvements made to the model, starting from a simple bigram model and gradually adding complex components like positional embeddings, attention mechanisms, and regularization techniques. Each change has brought incremental improvements to both training and validation loss, making the model more powerful and capable of capturing richer patterns in the data.

However, it is important to note that despite adding layers which theoretically should improve the model's performance, there were instances where the loss increased. This was particularly noticeable after adding certain layers like **dropout**. The increase in loss can be attributed to the fact that the model’s **hyperparameters** were not adjusted as the new layers were added. In practice, **dropout** randomly disables certain neurons, which can lead to higher loss values during training, especially when the model is not fine-tuned to account for these changes. The same hyperparameters were used throughout, without recalibrating them for each new architectural addition, which logically could lead to a temporary increase in loss.

Despite these challenges, the model showed clear improvements in its ability to capture complex relationships within the data as more layers were introduced.

---

## Final Model Results

The final model, which can be found in the **model** folder, was trained using the following hyperparameters:

- **BLOCK_SIZE**: 32
- **BATCH_SIZE**: 32
- **N_TRANSFORMER_BLOCKS**: 6
- **N_EMBED**: 128
- **NUM_HEADS**: 8
- **DROPOUT**: 0.2
- **LEARNING_RATE**: 3e-4
- **N_ITERS**: 8000
- **EVAL_INTERVAL**: 500
- **EVAL_ITERS**: 200

The model was trained on a dataset with:
- **Train text length**: 1,003,854 characters
- **Valid text length**: 111,540 characters
- **Number of model parameters**: 1,208,385

Here are the key results:

- **Step 0**: Train loss 4.68, Val loss 4.66
- **Step 500**: Train loss 2.37, Val loss 2.37
- **Step 1000**: Train loss 2.19, Val loss 2.21
- **Step 1500**: Train loss 2.07, Val loss 2.12
- **Step 2000**: Train loss 1.99, Val loss 2.06
- **Step 2500**: Train loss 1.93, Val loss 2.02
- **Step 3000**: Train loss 1.88, Val loss 1.98
- **Step 3500**: Train loss 1.84, Val loss 1.96
- **Step 4000**: Train loss 1.79, Val loss 1.93
- **Step 4500**: Train loss 1.77, Val loss 1.92
- **Step 5000**: Train loss 1.75, Val loss 1.89
- **Step 5500**: Train loss 1.72, Val loss 1.87
- **Step 6000**: Train loss 1.70, Val loss 1.86
- **Step 6500**: Train loss 1.68, Val loss 1.84
- **Step 7000**: Train loss 1.66, Val loss 1.83
- **Step 7500**: Train loss 1.65, Val loss 1.83
- **Step 7999**: Train loss 1.64, Val loss 1.81

### Observations:

- **Training Loss vs. Validation Loss**: Early in the training process, the model improves both on the training and validation datasets. However, after a certain point, the training loss continues to decrease, while the validation loss stagnates or slightly increases, suggesting the model is overfitting to the training data.

- **Possible Solutions**: To mitigate overfitting, the following strategies could be explored:
  - **Hyperparameter Tuning**: Adjusting the learning rate, batch size, or dropout rate could help in reducing overfitting.
  - **Early Stopping**: Monitoring validation loss and stopping training when it starts increasing can help prevent overfitting.
  - **Data Augmentation**: Adding more diverse data or using techniques like data augmentation could help improve generalization.

---

## 6. Future Improvements

While the current model demonstrates solid performance, there are several areas for improvement and further exploration. In this section, we outline some potential directions to enhance the model's capacity, efficiency, and generalization.

### 6.1 Hyperparameter Tuning

One of the main factors affecting the model's performance is the choice of hyperparameters. In the current implementation, the hyperparameters were kept constant across the different stages of the model's development, which likely contributed to some increase in loss (especially after adding regularization techniques like **dropout**). 

To improve performance, it would be beneficial to conduct a systematic hyperparameter search, adjusting:
- **Learning rate**: To find the optimal learning rate for convergence.
- **Batch size**: To strike the right balance between memory efficiency and model performance.
- **Dropout rate**: Adjusting the dropout rate to better prevent overfitting without harming performance.
- **Number of heads in attention**: Experimenting with different numbers of attention heads in the multi-head attention mechanism to see if more heads lead to better learning.

### 6.2 More Advanced Regularization Techniques

While **dropout** was introduced as a regularization technique, there are other methods worth exploring, including:
- **Weight decay (L2 regularization)**: Adding weight decay can help reduce overfitting by penalizing large weights in the network.
- **Layer-wise learning rate decay**: This technique allows different layers of the model to learn at different rates, improving training stability and potentially boosting performance.

### 6.3 Larger Dataset

Currently, the model is trained on a relatively small dataset (~1M characters), which might limit its ability to generalize well. Using a **larger dataset**, or fine-tuning on domain-specific data, would likely help the model better capture more complex patterns in the language.

---

## 7. Conclusion

In this project, we successfully implemented a **GPT-style language model from scratch**, leveraging the Transformer architecture to perform **character-level language modeling**. The model was progressively built, starting with a simple **bigram model** and gradually incorporating more complex components like **positional embeddings**, **multi-head self-attention**, and **dropout** to improve performance and generalization.

Despite some challenges, such as **overfitting** and the need for further **hyperparameter tuning**, the model demonstrated solid performance. The results from training and validation loss show that the model was able to learn meaningful patterns from the data, but there are still opportunities to refine the architecture and training process to further improve accuracy and generalization.

### Key Takeaways:
- **Progressive Development**: The model was incrementally built, starting with simpler components and gradually adding more complexity.
- **Overfitting**: The model experienced signs of overfitting, which could be addressed by tuning hyperparameters or incorporating additional regularization techniques.
- **Future Work**: Future improvements include hyperparameter tuning, exploring advanced Transformer variants, using pretrained models, and fine-tuning for specific NLP tasks.
