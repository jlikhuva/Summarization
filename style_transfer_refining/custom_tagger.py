'''
Defines a seq2seq model
for tagging words in a source
document as either include(1)
or not include (0)
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class SequenceTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=512, out_size=2):
        super(SequenceTagger, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.out = nn.Linear(hidden_dim, out_size)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        embeddings_packed = pack_padded_sequence(embeddings, lengths)
        lstm_out, _ = self.lstm(embeddings_packed)
        lstm_out_unpacked = pad_packed_sequence(lstm_out)[0]
        out_input = lstm_out_unpacked.view(-1, lstm_out_unpacked.shape[2])
        out = self.out(out_input)
        return F.log_softmax(out, dim=1)
