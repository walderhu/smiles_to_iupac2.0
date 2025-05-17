#!/bin/bash

set -e 

conda deactivate
conda env remove -n iu -y
conda env create -f /home/lipatovdn/SMILES_to_IUPAC/model/conda.yml
conda activate iu

cd /home/lipatovdn/model/ 
python3 /home/lipatovdn/model/_train.py  
cd -