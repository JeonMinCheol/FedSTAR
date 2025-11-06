model=mobilenet_v3
dcl=0.5
lbs=8192
jr=0.6
nb=31
nc=100
ls=5

# =================================================================== CIFAR-100 ==================================================================
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo APFL -lr 0.8
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo Ditto -lr 0.002
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo FedALA -lr 0.01
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo FedMTL -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo FedProto -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo FedRep -lr 0.003
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo MOON -lr 0.1
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo ablation -lps 1.0 -lpp 0.7 -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Cifar100 -m $model -algo FedSTAR -lps 1.0 -lpp 0.7 -lr 0.005
# ================================================================================================================================================

# =================================================================== FMNIST =====================================================================
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo APFL -lr 0.8
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo Ditto -lr 0.002
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo FedALA -lr 0.05
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo FedMTL -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo FedProto -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo FedRep -lr 0.003
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo MOON -lr 0.1
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo ablation -lps 0.3 -lpp $jr -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data FMNIST -m $model -algo FedSTAR -lps 1.0 -lpp 0.7 -lr 0.005
# =================================================================================================================================================

# =================================================================== Office-31 ===================================================================
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo APFL -lr 0.4
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo Ditto -lr 0.002
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo FedALA -lr 0.1
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo FedMTL -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo FedProto -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo FedRep -lr 0.003
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo MOON -lr 0.1
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo ablation -lps 1.0 -lpp $jr -lr 0.005
# python -u main.py -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb $nb -ls $ls -data Office-31 -m $model -algo FedSTAR -lps 1.0 -lpp 0.7 -lr 0.005
# =================================================================================================================================================