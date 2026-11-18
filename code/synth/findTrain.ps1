"Adam 3:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 3 --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"Adam 1:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 1 --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"Adamax 3:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 3 --optim Adamax --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"Adamax 1:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 1 --optim Adamax --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"Adagrad 3:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 3 --optim Adagrad --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"Adagrad 1:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 1 --optim Adagrad --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"" >> out.txt
"GD 3:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 3 --optim GD --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"GD 1:" >> out.txt
python.exe .\main.py --lr 0.001 --lam 0.05 --depth 1 --optim GD --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
"" >> out.txt