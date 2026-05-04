class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, output_size, hidden_dim, n_layers, dropout_prob):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Set to False if you don't want to train the embeddings

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, hidden = self.gru(embedded)
        gru_output = self.dropout(gru_output[:, -1, :])  # Only take the output of the last time step
        output = self.fc(gru_output)
        return output