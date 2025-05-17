<h1>Reaction & Molecule Encoder</h1>

<h2>Installation</h2>
First install Pytorch compatible with your system and GPU driver version - https://pytorch.org/get-started/locally/.
<br>
Then run pip installation:

```
	python3 -m pip install .
```

<h2>Usage</h2>
<h3>Load weights for tuning</h3>

- Choose model and checkpoint you want to use from "models" directory. You can find training logs of these models in "logs" directory and watch them with tensorboard.
- Load model:
  ```
  import RMolEncoder as rme
  
  pretrained_model = rme.PretrainModel.load_from_checkpoint("models/768d_12h_8ls_shared-True/step=1250000_ep=63_valid_loss=0.049004.ckpt")    
  ```
- Create your model with rme submodule inside:
  ```
  class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = rme.RMolEmbedder(
                            encoder_hparams = dict(
                                    d_model=768, 
                                    n_in_head=12,
                                    num_in_layers=8, 
                                    shared_weights=True
                            ),
                            mlp_hparams = dict(
                                     in_dim=768,
                                     dims=[512, 256], drop=0.1, 
                                     mlp_activation=torch.nn.GELU(), 
                            )
                        )
        
        self.my_linear_head = torch.nn.Linear(256, 1)

    def forward(self, x):
        mlp_emb, cls_emb = self.encoder(x)
        pred = self.my_linear_head(mlp_emb)
        return pred
  ```
- Create your model and load weights from pretrained embedder:
```
mymodel = MyModel()
mymodel.encoder.load_state_dict(pretrained_model.embedder.state_dict())

# Optionally freeze pretrained weights
mymodel.encoder.requires_grad_(False)
```
- Tune it as you want

<h3>How to feed data</h3>

Use rme package's Dataset and Dataloader to create batches of data. Dataset can process both smiles strings and chython containers.
  ```
  ds = rme.RxnMolDataset([
      "CCO",              # molecule smiles
      "CCN>>CC.N",        # reaction smiles
      smiles("O"),        # molecule container
      smiles("C.O>>C=O"), # reaction container
  ], 
  # Optional target for training. It can be torch tensor or dictionary so you can pass several named target values
  target = torch.rand(4)
  )
  
  x, y = ds[0]
  ```
Create dataloader to make batches:
```
dl = rme.make_dataloader(ds, batch_size=2, shuffle=False, drop_last=False)
```
Call the model:
```
# Take batch
for x, y in dl:
    break
    
mymodel(x)
>>> tensor([[ 0.0763],
        [-0.1222]], grad_fn=<AddmmBackward0>)
```

































