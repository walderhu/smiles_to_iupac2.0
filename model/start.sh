
cd /home/lipatovdn/smiles_to_iupac2.0/model
rm -rf data-info.json  klasterisation.log
screen -dmS klaster python3 "/home/lipatovdn/smiles_to_iupac2.0/model/__klaster.py"
