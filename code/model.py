import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=4, hidden_size=512, num_layers=4):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer_encoder(embedded)
        return self.fc(output)