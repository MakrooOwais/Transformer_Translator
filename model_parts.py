import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    """
    Applies layer normalization to the input tensor.

    Layer normalization normalizes the inputs across the features dimension for each
    data point in the batch. It ensures that the inputs to each layer have a mean of 0
    and a standard deviation of 1, which helps in stabilizing the learning process.

    Attributes:
        eps (float): A small value to prevent division by zero during normalization.
        alpha (nn.Parameter): A learnable parameter that scales the normalized input.
        bias (nn.Parameter): A learnable parameter that shifts the normalized input.

    Args:
        features (int): The number of features in the input tensor.
        eps (float, optional): A small value to prevent division by zero. Default is 1e-6.
    """

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            torch.Tensor: The layer-normalized output tensor of the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements a feedforward neural network block used in Transformer models.

    This block consists of two linear transformations with a ReLU activation in between,
    followed by dropout for regularization. It is typically used in the position-wise feedforward
    part of Transformer models.

    Attributes:
        linear_1 (nn.Linear): The first linear transformation layer.
        dropout (nn.Dropout): The dropout layer for regularization.
        linear_2 (nn.Linear): The second linear transformation layer.

    Args:
        d_model (int): The dimension of the input tensor.
        d_ff (int): The dimension of the hidden layer.
        dropout (float): The dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        """
        Forward pass for the feedforward block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """
    Converts input tokens to dense embeddings suitable for input to a transformer model.

    This class uses an embedding layer to convert input token IDs to dense vectors.
    The vectors are then scaled by the square root of the model dimension to maintain
    the variance of the input embeddings.

    Attributes:
        d_model (int): The dimension of the embedding vectors.
        vocab_size (int): The size of the vocabulary.
        embedding (nn.Embedding): The embedding layer that converts token IDs to dense vectors.

    Args:
        d_model (int): The dimension of the embedding vectors.
        vocab_size (int): The size of the vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass for converting input token IDs to embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len) containing token IDs.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model) containing embeddings.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings to provide a sense of order
    for the sequence data in transformer models.

    This class computes a fixed positional encoding matrix using sine and cosine functions
    of different frequencies and adds it to the input embeddings. This helps the transformer
    model to learn the position of each token in the sequence.

    Attributes:
        d_model (int): The dimension of the embedding vectors.
        seq_len (int): The length of the input sequences.
        dropout (nn.Dropout): The dropout layer for regularization.
        pe (torch.Tensor): The positional encoding matrix.

    Args:
        d_model (int): The dimension of the embedding vectors.
        seq_len (int): The length of the input sequences.
        dropout (float): The dropout probability.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)

        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass to add positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model) with positional encodings added.
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.

    This class adds the output of a sublayer to its input (residual connection) and applies
    dropout and layer normalization. Residual connections help in training deep networks
    by allowing gradients to flow through the network directly.

    Attributes:
        dropout (nn.Dropout): The dropout layer for regularization.
        norm (LayerNormalization): The layer normalization module.

    Args:
        features (int): The number of features in the input tensor.
        dropout (float): The dropout probability.
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Forward pass for the residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).
            sublayer (nn.Module): The sublayer to be applied to the normalized input.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input, with residual connection applied.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    Implements the multi-head attention mechanism used in Transformer models.

    This class allows the model to jointly attend to information from different
    representation subspaces at different positions. It consists of multiple
    attention heads, each of which performs scaled dot-product attention in parallel.

    Attributes:
        d_model (int): The dimension of the input and output embeddings.
        h (int): The number of attention heads.
        d_k (int): The dimension of the vector seen by each attention head.
        w_q (nn.Linear): Linear layer to project inputs to query vectors.
        w_k (nn.Linear): Linear layer to project inputs to key vectors.
        w_v (nn.Linear): Linear layer to project inputs to value vectors.
        w_o (nn.Linear): Linear layer to project concatenated attention outputs.
        dropout (nn.Dropout): Dropout layer for regularization.
        attention_scores (torch.Tensor): Tensor to store attention scores for visualization.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        h (int): The number of attention heads.
        dropout (float): The dropout probability.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Computes scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, h, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch, h, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch, h, seq_len, d_k).
            mask (torch.Tensor): Mask tensor to prevent attention to certain positions.
            dropout (nn.Dropout): Dropout layer for regularization.

        Returns:
            torch.Tensor: Output tensor of shape (batch, h, seq_len, d_k).
            torch.Tensor: Attention scores of shape (batch, h, seq_len, seq_len).
        """
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for the multi-head attention block.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, seq_len, d_model).
            k (torch.Tensor): Key tensor of shape (batch, seq_len, d_model).
            v (torch.Tensor): Value tensor of shape (batch, seq_len, d_model).
            mask (torch.Tensor): Mask tensor to prevent attention to certain positions.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Implements a single encoder block used in Transformer models.

    This block consists of a multi-head self-attention mechanism followed by
    a position-wise feedforward neural network. Residual connections and
    layer normalization are applied around each of these sub-layers.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention module.
        feed_forward_block (FeedForwardBlock): The feedforward neural network module.
        residual_connections (nn.ModuleList): A list containing the residual connection modules.

    Args:
        features (int): The dimension of the input and output embeddings.
        self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention module.
        feed_forward_block (FeedForwardBlock): The feedforward neural network module.
        dropout (float): The dropout probability.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Forward pass for the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).
            src_mask (torch.Tensor): Source mask tensor to prevent attention to certain positions.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Implements a single decoder block used in Transformer models.

    This block consists of a masked multi-head self-attention mechanism,
    a multi-head cross-attention mechanism, and a position-wise feedforward
    neural network. Residual connections and layer normalization are applied
    around each of these sub-layers.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The masked multi-head self-attention module.
        cross_attention_block (MultiHeadAttentionBlock): The multi-head cross-attention module.
        feed_forward_block (FeedForwardBlock): The feedforward neural network module.
        residual_connections (nn.ModuleList): A list containing the residual connection modules.

    Args:
        features (int): The dimension of the input and output embeddings.
        self_attention_block (MultiHeadAttentionBlock): The masked multi-head self-attention module.
        cross_attention_block (MultiHeadAttentionBlock): The multi-head cross-attention module.
        feed_forward_block (FeedForwardBlock): The feedforward neural network module.
        dropout (float): The dropout probability.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the decoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch, seq_len, features).
            src_mask (torch.Tensor): Source mask tensor to prevent attention to certain positions in the source sequence.
            tgt_mask (torch.Tensor): Target mask tensor to prevent attention to certain positions in the target sequence.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Implements the decoder part of the Transformer model.

    The decoder consists of multiple decoder blocks, each containing a masked multi-head
    self-attention mechanism, a multi-head cross-attention mechanism, and a position-wise
    feedforward neural network. A final layer normalization is applied to the output.

    Attributes:
        layers (nn.ModuleList): A list containing the decoder blocks.
        norm (LayerNormalization): The layer normalization module.

    Args:
        features (int): The dimension of the input and output embeddings.
        layers (nn.ModuleList): A list containing the decoder blocks.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, tgt_seq_len, features).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch, src_seq_len, features).
            src_mask (torch.Tensor): Source mask tensor to prevent attention to certain positions in the source sequence.
            tgt_mask (torch.Tensor): Target mask tensor to prevent attention to certain positions in the target sequence.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Implements the projection layer used in Transformer models for generating output.

    This layer projects the decoder output from the model's hidden dimension to the vocabulary size,
    producing logits for each token in the output sequence.

    Attributes:
        proj (nn.Linear): Linear projection layer.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        vocab_size (int): The size of the vocabulary, i.e., the number of output tokens.
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        """
        Forward pass for the projection layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, vocab_size).
        """
        return self.proj(x)


class Transformer(nn.Module):
    """
    Implements the Transformer model consisting of an encoder, decoder, and projection layer.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): Source input embeddings module.
        tgt_embed (InputEmbeddings): Target input embeddings module.
        src_pos (PositionalEncoding): Source positional encoding module.
        tgt_pos (PositionalEncoding): Target positional encoding module.
        projection_layer (ProjectionLayer): The projection layer for output.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): Source input embeddings module.
        tgt_embed (InputEmbeddings): Target input embeddings module.
        src_pos (PositionalEncoding): Source positional encoding module.
        tgt_pos (PositionalEncoding): Target positional encoding module.
        projection_layer (ProjectionLayer): The projection layer for output.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): Source tensor of shape (batch, src_seq_len).
            src_mask (torch.Tensor): Source mask tensor to prevent attention to certain positions in the source sequence.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch, src_seq_len, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        """
        Decodes the target sequence based on encoder output.

        Args:
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch, src_seq_len, d_model).
            src_mask (torch.Tensor): Source mask tensor to prevent attention to certain positions in the source sequence.
            tgt (torch.Tensor): Target tensor of shape (batch, tgt_seq_len).
            tgt_mask (torch.Tensor): Target mask tensor to prevent attention to certain positions in the target sequence.

        Returns:
            torch.Tensor: Decoded tensor of shape (batch, tgt_seq_len, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Projects the decoder output to logits over the vocabulary.

        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch, tgt_seq_len, d_model).

        Returns:
            torch.Tensor: Tensor of shape (batch, tgt_seq_len, vocab_size) containing logits.
        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    Builds a Transformer model architecture.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum sequence length of the source inputs.
        tgt_seq_len (int): Maximum sequence length of the target inputs.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder layers. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimensionality of the feedforward layers. Defaults to 2048.

    Returns:
        Transformer: Constructed Transformer model.
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
