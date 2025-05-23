import sentencepiece as spm
import tempfile, logging
import pandas as pd, os, sys
from os.path import join, abspath, basename
from io import StringIO

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('sentencepiece').setLevel(logging.CRITICAL)

dirname = abspath(join(os.getcwd(), '__unique_data__'))
token_dirname = abspath(join(os.getcwd(), '__tokens__'))

if os.path.exists(token_dirname):
    import shutil
    shutil.rmtree(token_dirname)

os.makedirs(token_dirname)

def batch_reader(file_path, batch_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        batch = []
        for line in f:
            batch.append(line.rstrip('\n'))
            if len(batch) == batch_size:
                yield '\n'.join([header] + batch)
                batch = []
        if batch: 
            yield '\n'.join([header] + batch)
            

def handler(filename):
    filepath = join(dirname, filename)
    model_filenpath = join(token_dirname, filename.removesuffix('.csv'))
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as f: 
        for batch in batch_reader(filepath, batch_size=2000):  
            batch_df = pd.read_csv(StringIO(batch), sep=';')
            smiles_batch = batch_df['IUPAC Name'].dropna().tolist()
            f.write("\n".join(smiles_batch) + "\n")  
        
        f.flush()  
        spm.SentencePieceTrainer.train(
            input=f.name,
            model_prefix=model_filenpath,
            vocab_size=500,
            model_type="bpe",
            split_digits=True,  
            treat_whitespace_as_suffix=False,
            normalization_rule_name="identity"  
        )
    
        

if __name__ == "__main__":
    from Kiki._send_msg import send_msg
    try:
        send_msg(f'Старт {basename(__file__)}', delete_after=5)

        for f in os.listdir(dirname): 
            if f.endswith('.csv'): handler(f)

        send_msg(f'Конец {basename(__file__)}')
    except Exception as e:
        import traceback
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        send_msg(f"Ошибка в {basename(__file__)}:\n{tb_str}")
        print(tb_str)
    finally:
        sys.exit(0)