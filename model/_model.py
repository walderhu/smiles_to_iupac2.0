from os.path import exists

import RMolEncoder as rme
import torch
import torch.nn as nn
from chytorch.utils.data import collate_encoded_reactions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_path = "pretrained_encoder.pth"
if not exists(pretrained_path):
    pretrained_path = None


class ChemLM(nn.Module):
    def __init__(self, chem_encoder_params: dict, decoder_params: dict, num_layers: int,
                 vocab_size: int, *, chem_encoder_pretrained_path: str = None):
        super().__init__()
        self.encoder = rme.ChemEncoder(**chem_encoder_params)
        if chem_encoder_pretrained_path:
            state_dict = torch.load(chem_encoder_pretrained_path, map_location=torch.device(device))
            self.encoder.load_state_dict(state_dict)

        self.embedding = nn.Embedding(vocab_size, decoder_params["d_model"])
        pe = self.calculate_pos_enc(4096, decoder_params["d_model"]) # 🔥 self.calculate_pos_enc(2048, ...
        self.pe = nn.parameter.Parameter(pe, requires_grad=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**decoder_params), num_layers=num_layers)

        self.proj = nn.Linear(chem_encoder_params["d_model"], decoder_params["d_model"])
        self.out_linear = nn.Linear(decoder_params["d_model"], vocab_size)

        for l in self.proj, self.out_linear:
            nn.init.xavier_uniform_(l.weight, gain=1.0)
            nn.init.zeros_(l.weight)

    def calculate_pos_enc(self, seq, dim):
        poses = [[pos / (10000**(2 * (j // 2) / dim))
                  for j in range(dim)] for pos in range(seq)]
        pos_enc = torch.FloatTensor(poses)
        pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def forward(self, encx, tgt):
        cls_emb, atom_embs = self.encoder(encx)
        enc_out = torch.cat([cls_emb.unsqueeze(1), atom_embs], dim=1)  # b, 1+atoms, dim
        enc_out = self.proj(enc_out)

        seql = tgt.shape[1]
        causal_mask = torch.triu(torch.full((seql, seql), -torch.inf, device=tgt.device), diagonal=1)
        atoms_pad_mask: bool = (encx.atoms == 0)

        x = self.embedding(tgt)
        x += self.pe[:seql]
        x = self.decoder(x, enc_out, tgt_mask=causal_mask, memory_key_padding_mask=atoms_pad_mask, tgt_is_causal=True)
        x = self.out_linear(x)  # d_model -> vocab
        probs = nn.functional.softmax(x, dim=-1)
        return probs


def tokenize(s: str, add_spezial_tokens: bool = False):
    tokens = [ord(c) for c in s]
    if add_spezial_tokens:
        tokens = [1] + tokens + [2]
    return torch.IntTensor(tokens)


def make_fix_len_collate(max_len):
    def collate_with_target(batch):
        x = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        Y = torch.stack([nn.functional.pad(y, pad=(0, 1 + max_len - y.size(0)), value=0) for y in ys])
        return collate_encoded_reactions(x), Y
    return collate_with_target


def Loss(p, y):
    l = - torch.log(p + 1e-7)
    l *= nn.functional.one_hot(y.long(), num_classes=p.shape[-1]).to(p.dtype)
    l = l.sum(-1).mean()
    return l


def batch_reader(filepath, batch_size=5e4):
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        batch = []
        for line in f:
            batch.append(line.rstrip('\n'))
            if len(batch) == batch_size:
                yield '\n'.join([header] + batch)
                batch = []
        if batch:
            yield '\n'.join([header] + batch)


__all__ = ['ChemLM', 'batch_reader', 'tokenize', 'make_fix_len_collate', 'Loss']
