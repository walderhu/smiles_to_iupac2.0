import difflib
import os
from csv import QUOTE_ALL
from os.path import basename, exists, join

import pandas as pd
import RMolEncoder as rme
import torch
import torch.nn as nn
from chytorch.utils.data import collate_encoded_reactions
from tqdm import tqdm

from _model import ChemLM, Loss, make_fix_len_collate, tokenize

from statistics import mean
import os
from io import StringIO
from os.path import exists, join, basename
from tqdm import tqdm
from typing import *
import pandas as pd
import RMolEncoder as rme
import torch
import torch.nn as nn
from __features import *

from _model import ChemLM, Loss, batch_reader, make_fix_len_collate, tokenize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestModel:
    def __init__(self, model):
        self.model = model
        self.DS_SEQ_LEN = None

    def evaluate(self, dataloader, max_len):
        self.model.eval()
        total_loss, total_tokens, correct_tokens, exact_matches = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in dataloader:
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
        try:
            x = self.encode(smiles)
            if not x:  
                return ""
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
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {str(e)}")
            return ""  
        
    
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
        loss = metrics['loss']
        token_accuracy = metrics['token_accuracy']
        exact_accuracy = metrics['exact_accuracy']
        return loss, token_accuracy, exact_accuracy


    def run_tests(self, test_df, filename) -> None:
        filename = join(predicted_dirname, f'predicted_{basename(filename)}')
        results = []
        for i, row in test_df.iterrows():
            pred = self.predict_iupac(row['SMILES'])
            seq = difflib.SequenceMatcher(None, pred, row['IUPAC Name'])
            results.append({
                'Correct': pred == row['IUPAC Name'],
                'Accuracy': f"{seq.ratio():.2f}",
                'SMILES': row['SMILES'],
                'True IUPAC': row['IUPAC Name'],
                'Predicted IUPAC': pred
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False, sep=';', escapechar='\\',
                          quoting=QUOTE_ALL, quotechar='"')
        # incorrect = results_df[~results_df['Correct']]



def meta(t_loss: list, t_token_accuracy: list, t_exact_accuracy: list) -> str:
    m_loss = f'{mean(t_loss):.4f}' if t_loss else '0.0000'
    m_token_accuracy = f'{mean(t_token_accuracy):.4f}' if t_token_accuracy else '0.0000'
    m_exact_accuracy = f'{mean(t_exact_accuracy):.4f}' if t_exact_accuracy else '0.0000'
    return f'Loss: {m_loss} | Token Acc: {m_token_accuracy} | Exact Acc: {m_exact_accuracy}'

def main(filename):
    model = ChemLM(chem_encoder_params=dict(d_model=512, n_in_head=8, num_in_layers=8, shared_weights=True),
                   decoder_params=dict(d_model=512, nhead=8, dim_feedforward=4*512, dropout=0.1,
                                       activation=nn.functional.gelu, batch_first=True, norm_first=True, bias=True),
                   num_layers=4, vocab_size=128, chem_encoder_pretrained_path=pretrained_path)
    
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()
    test_data_files = [join(test_dirname, file) for file in os.listdir(test_dirname)]
    t_loss, t_token_accuracy, t_exact_accuracy = [], [], []
    batch_size = 256    
    tester = TestModel(model)  
    
    for file in test_data_files:
        total_lines = count_lines(file)  
        with tqdm(total=total_lines, desc=meta(t_loss, t_token_accuracy, t_exact_accuracy), 
                 dynamic_ncols=True) as tq:
            
            for num_batch, batch in enumerate(batch_reader(file, batch_size), start=1):
                try:
                    df: pd.DataFrame = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
                    
                    loss, token_accuracy, exact_accuracy = tester.run_evaluation(df)
                    t_loss.append(loss)
                    t_token_accuracy.append(token_accuracy)
                    t_exact_accuracy.append(exact_accuracy)
                    # tester.run_tests(df, file)
                    tq.set_description(meta(t_loss, t_token_accuracy, t_exact_accuracy))
                    tq.update(batch_size)
                except Exception as e:
                    print(f"Error processing batch {num_batch} from file {file}: {str(e)}")
                    continue


if __name__ == '__main__':
    test_dirname = '__test__'
    if not exists(test_dirname):
        raise FileNotFoundError(f"Нет папки с тестовой выборкой по пути: {test_dirname}")

    pretrained_path = "pretrained_encoder.pth"
    if not exists(pretrained_path):
        raise FileNotFoundError(f"Предобученный файл енкодера не найден по пути: {pretrained_path}")

    filename = "model.pth"
    if (filename is None) or (not exists(filename)):
        raise FileNotFoundError(f"Предобученный файл модели не найден по пути: {filename}")

    predicted_dirname = '__predicted__'
    os.makedirs(predicted_dirname, exist_ok=True)
    main(filename)
