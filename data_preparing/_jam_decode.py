import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="model.model")

iupac = "2,4-dichlorophenol"
tokens = sp.encode(iupac, out_type=str)
print(tokens)  # ['▁', '2', ',', '4', '-', 'dichlorophenol']

text = sp.decode(tokens)
print(text) # 2,4-dichlorophenol


sp = spm.SentencePieceProcessor(model_file="model.model")
print(sp.encode("C6H12O6", out_type=str))  # ['▁', 'C', '6', 'H', '1', '2', 'O', '6']