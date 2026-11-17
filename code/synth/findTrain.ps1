python.exe .\main.py --lr 0.001 --lam 0.01 --depth 3 --niters 500000 2>&1 |
    ForEach-Object { "$_".Trim() } |
    Where-Object { $_ -ne "" } |
    Add-Content out.txt
#python.exe .\main.py --lr 0.001 --lam 0.05 --depth 1 >> out.txt
#"" >> out.txt
#python.exe .\main.py --lr 0.0001 --lam 0.05 --depth 3 >> out.txt
#"" >> out.txt
#python.exe .\main.py --lr 0.0001 --lam 0.05 --depth 1 >> out.txt
#"" >> out.txt
#python.exe .\main.py --lr 0.00001 --lam 0.05 --depth 3 >> out.txt
#"" >> out.txt
#python.exe .\main.py --lr 0.00001 --lam 0.05 --depth 1 >> out.txt
#"" >> out.txt
#python.exe .\main.py --lr 0.000001 --lam 0.05 --depth 3 >> out.txt
#"" >> out.txt
#python.exe .\main.py --lr 0.000001 --lam 0.05 --depth 1 >> out.txt