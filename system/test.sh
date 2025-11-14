model=mobilenet_v3
dcl=0.1
lbs=8192
jr=0.3
nc=5
ls=5
gr=200

# =================================================================== CIFAR-100 ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo APFL -lr 0.8
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo Ditto -lr 0.002
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedALA -lr 0.01
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedMTL -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedProto -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedRep -lr 0.003
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo MOON -lr 0.1
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo ablation -lps 1.0 -lpp 0.7 -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedSTAR -lps 1.0 -lpp 0.7 -lr 0.005
# =================================================================================================================================================

# =================================================================== fmnist ======================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo APFL -lr 0.8
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo Ditto -lr 0.002
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedALA -lr 0.05
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedMTL -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedProto -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedRep -lr 0.003
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo MOON -lr 0.1
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo ablation -lps 0.3 -lpp 0.6 -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedSTAR -lps 1.0 -lpp 0.7 -lr 0.005
# =================================================================================================================================================

# =================================================================== Office-31 ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo APFL -lr 0.4
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo Ditto -lr 0.002
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedALA -lr 0.1
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedMTL -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedProto -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedRep -lr 0.003
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo MOON -lr 0.1
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo ablation -lps 1.0 -lpp 0.6 -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedSTAR -lps 1.0 -lpp 0.7 -lr 0.005
# =================================================================================================================================================

# =================================================================== DomainNet ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo APFL -lr 0.8
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo Ditto -lr 0.002
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedALA -lr 0.05
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedMTL -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedProto -lr 0.005
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedRep -lr 0.003
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo MOON -lr 0.1
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo ablation -lps 0.3 -lpp 0.7 -lr 0.003
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedSTAR -lps 1.0 -lpp 0.6 -lr 0.005
# =================================================================================================================================================
