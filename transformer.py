import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, vocab_size=10000, max_seq_length=512):
        """
        Standard Transformer Model Constructor
        Args:
            d_model: Embedding size
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Hidden size of the feed-forward network
            dropout: Dropout rate
            vocab_size: Vocabulary size
            max_seq_length: Maximum sequence length
        """
        super(Transformer, self).__init__()
        
        # Embedding layers for input and output tokens
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.output_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # Transformer layers
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        Forward pass for Transformer
        Args:
            src: Source input tokens, shape (batch_size, src_seq_len)
            tgt: Target input tokens, shape (batch_size, tgt_seq_len)

        Returns:
            Logits, shape (batch_size, tgt_seq_len, vocab_size)
        """
        batch_size, src_seq_len = src.size()
        _, tgt_seq_len = tgt.size()

        # Embedding and add positional encoding
        src_emb = self.input_embedding(src) + self.positional_encoding[:, :src_seq_len, :]
        tgt_emb = self.output_embedding(tgt) + self.positional_encoding[:, :tgt_seq_len, :]

        # Transformer forward pass
        src_emb = src_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)
        tgt_emb = tgt_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)

        output = self.transformer(src_emb, tgt_emb)

        # Back to (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)

        # Final output layer
        logits = self.fc_out(output)
        return logits
