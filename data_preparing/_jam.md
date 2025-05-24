screen -dmS encode python3 "/home/lipatovdn/SMILES_to_IUPAC/data_preparing/_jam_encode.py"


screen -dmS token python3 "/home/lipatovdn/smiles_to_iupac2.0/data_preparing/__tiffany.py"
echo "screen -dmS tokenize python3 file.py" | at now + 3 hours
(sleep 3h && screen -dmS tokenize python3 file.py) &
(sleep 10s && echo hello) &
nohup bash -c "sleep $((3)) && echo 'hello'" &



