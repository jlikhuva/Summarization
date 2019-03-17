'''
Takes in the latent representation from the
style transfer module and produces
the abstractive summary
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class STDecoder(nn.Module):
    """
        Module for decoding,  given the latent
        representation of the 'style transfer' module.
        Portions borrowed from CS224 A4.
    """
    def __init__(self, vocab_size, embed_size=100, hidden_size=512, dropout_rate=0.2):
        super(STDecoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size, bias=True)

        # Global Attention
        self.h_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.att_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size, bias=False)

        self.target_vocab_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, latent_state, dec_init_state, target_padded, enc_masks=None):
        target_padded = target_padded[:-1]
        dec_state = dec_init_state
        batch_size = latent_state.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size)

        combined_outputs = []
        enc_hiddens_proj = self.att_projection(latent_state)
        Y = self.embeddings(target_padded)
        time_chunks = torch.split(Y, 1)
        for chunk in time_chunks:
            Y_t  = torch.squeeze(chunk, dim=0)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            dec_state, o_t, e_t = self.step(
                Ybar_t, dec_state, latent_state,
                enc_hiddens_proj, enc_masks
            )
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs

    def step(self, Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.bmm(dec_hidden.unsqueeze(dim=1), enc_hiddens_proj.permute(0, 2, 1))
        e_t = e_t.squeeze(dim=1)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
        alpha_t = F.softmax(e_t, dim=-1)
        a_t = torch.bmm(alpha_t.unsqueeze(dim=1), enc_hiddens).squeeze(dim=1)
        U_t = torch.cat((dec_hidden, a_t), dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        combined_output = O_t
        return dec_state, combined_output, e_t
