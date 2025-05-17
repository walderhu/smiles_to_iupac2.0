import os
from csv import QUOTE_ALL
from os.path import exists, join, basename
import difflib

import pandas as pd
import RMolEncoder as rme
import torch
import torch.nn as nn
from chytorch.utils.data import collate_encoded_reactions
from tqdm import tqdm

from _model import ChemLM, Loss, make_fix_len_collate, tokenize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predicted_dirname = '__predicted__'
os.makedirs(predicted_dirname, exist_ok=True)

class TestModel:
    def __init__(self, model):
        self.model = model
        self.DS_SEQ_LEN = None

    def evaluate(self, dataloader, max_len):
        self.model.eval()
        total_loss, total_tokens, correct_tokens, exact_matches = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Evaluating"):
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    p = self.model(x, y[:, :-1])
                    loss = Loss(p, y[:, 1:])

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size

                preds = torch.argmax(p, dim=-1)
                targets = y[:, 1:]

                mask = (targets != 0)
                total_tokens += mask.sum().item()
                correct_tokens += ((preds == targets) & mask).sum().item()

                for i in range(preds.size(0)):
                    pred_seq = preds[i][mask[i]]
                    target_seq = targets[i][mask[i]]
                    if torch.all(pred_seq == target_seq):
                        exact_matches += 1

        avg_loss = total_loss / len(dataloader.dataset)
        token_accuracy = correct_tokens / total_tokens
        exact_accuracy = exact_matches / len(dataloader.dataset)
        return {'loss': avg_loss,
                'token_accuracy': token_accuracy,
                'exact_accuracy': exact_accuracy}

    def encode(self, *smiles):
        ds = rme.dataset.RxnMolDataset(smiles, [''] * len(smiles))
        return [x for x, _ in ds]

    def predict_iupac(self, smiles, max_len=None):
        if max_len is None:
            max_len = self.DS_SEQ_LEN

        self.model.eval()
        x = self.encode(smiles)
        x = collate_encoded_reactions(x).to(device)
        current_token = torch.tensor([[1]], device=device)
        with torch.no_grad():
            for _ in range(max_len):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    p = self.model(x, current_token)
                next_token = torch.argmax(p[:, -1, :], dim=-1, keepdim=True)
                if next_token.item() == 2:
                    break
                current_token = torch.cat([current_token, next_token], dim=1)

        tokens = current_token[0].cpu().tolist()[1:]
        iupac = ''.join(chr(t) for t in tokens if t not in [1, 2])
        return iupac

    def run_evaluation(self, test_df: pd.DataFrame):
        MAX_SEQ_LEN = test_df["IUPAC Name"].apply(len).max()
        self.DS_SEQ_LEN = MAX_SEQ_LEN + 3

        test_smis = list(test_df['SMILES'])
        test_seqs = [tokenize(test_df.iloc[i]["IUPAC Name"], add_spezial_tokens=True)
                     for i in range(test_df.shape[0])]
        test_ds = rme.dataset.RxnMolDataset(test_smis, test_seqs)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128,
                                              collate_fn=make_fix_len_collate(self.DS_SEQ_LEN),
                                              shuffle=False, drop_last=False)
        metrics = self.evaluate(test_dl, self.DS_SEQ_LEN)
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")
        print(f"Exact Match Accuracy: {metrics['exact_accuracy']:.4f}")

    def run_tests(self, test_df, filename='__predictions.csv'):
        results = []
        for i, row in test_df.iterrows():
            pred = self.predict_iupac(row['SMILES'])
            seq = difflib.SequenceMatcher(None, pred, row['IUPAC Name'])
            results.append({
                            'Correct': pred == row['IUPAC Name'],
                            'Accuracy' : f"{seq.ratio():.2f}",
                            'SMILES': row['SMILES'],
                            'True IUPAC': row['IUPAC Name'],
                            'Predicted IUPAC': pred
                            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False, sep=';', escapechar='\\',
                          quoting=QUOTE_ALL, quotechar='"')

        incorrect = results_df[~results_df['Correct']]
        print(f"\nIncorrect predictions ({len(incorrect)}/{len(results_df)}):")
        print(incorrect[['SMILES', 'True IUPAC', 'Predicted IUPAC']].head(10))


def main():
    pretrained_path = "pretrained_encoder.pth"
    filename = "model.pth"

    if not exists(pretrained_path):
        raise FileNotFoundError(f"Предобученный файл енкодера не найден по пути: {pretrained_path}")
    if (filename is None) or (not exists(filename)):
        raise FileNotFoundError(f"Предобученный файл модели не найден по пути: {filename}")

    model = ChemLM(chem_encoder_params=dict(d_model=512, n_in_head=8, num_in_layers=8, shared_weights=True),
                   decoder_params=dict(d_model=512, nhead=8, dim_feedforward=4 * 512, dropout=0.1,
                                       activation=nn.functional.gelu, batch_first=True, norm_first=True, bias=True),
                   num_layers=4, vocab_size=128, chem_encoder_pretrained_path=pretrained_path)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()

    dirname = '__test__'
    test_data_files = [join(dirname, file) for file in os.listdir(dirname)]
    for file in test_data_files:
        test_df = pd.read_csv(file, delimiter=';', encoding='utf-8', nrows=10).dropna()
        tester = TestModel(model)
        tester.run_evaluation(test_df)
        
        filename = join(predicted_dirname, f'predicted_{basename(file)}')
        print(filename)
        tester.run_tests(test_df, filename)


if __name__ == '__main__':
    main()
