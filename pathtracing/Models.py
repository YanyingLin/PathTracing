import torch
import torch.nn as nn
import numpy as np
from pathtracing.Layers import EncoderLayer, DecoderLayer

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)

    return padding_mask


def get_subsequent_mask(seq):

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_token_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_token_emb = nn.Embedding(n_src_vocab, d_token_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_token_vec),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        enc_output = self.src_token_emb(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_token_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_token_emb = nn.Embedding(
            n_tgt_vocab, d_token_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_token_vec),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad.byte() + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_token_emb(tgt_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class PathTracing(nn.Module):

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_token_vec=512, embedding_dim=512, d_inner=2048,
            L_layers=6, dimension=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_token_vec=d_token_vec, d_model=embedding_dim, d_inner=d_inner,
            n_layers=L_layers, n_head=dimension, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_token_vec=d_token_vec, d_model=embedding_dim, d_inner=d_inner,
            n_layers=L_layers, n_head=dimension, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_token_prj = nn.Linear(embedding_dim, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_token_prj.weight)

        assert embedding_dim == d_token_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            self.tgt_token_prj.weight = self.decoder.tgt_token_emb.weight
            self.x_logit_scale = (embedding_dim ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            self.encoder.src_token_emb.weight = self.decoder.tgt_token_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_token_prj(dec_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2))
