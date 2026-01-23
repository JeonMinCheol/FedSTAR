model=mobilenet_v3
dcl=0.1
lbs=256
jr=0.3
ls=3
nc=100
gr=300

for seed in 42 43 44; do 
# =================================================================== CIFAR-100 ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedSTAR -lr 0.4 -alr 0.01 -sas 10 -sac 1.0 -dr 0.05 -udg True -uf True -ut True -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo Ditto -lr 0.15 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedALA -lr 0.5 -et 1.0 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedRep -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedMTL -lr 0.01 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedProto -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedPAC -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedTGP -lr 0.01 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================

# =================================================================== DomainNet ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedSTAR -lr 0.015 -alr 0.01 -al 1.0 -sas 10 -sac 1.0 -dr 0.05 -udg True -uf True -ut True -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedALA -lr 0.2 -et 1.0 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedPAC -lr 0.15 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedMTL -lr 0.005 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedProto -lr 0.005 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo Ditto -lr 0.005 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedRep -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedTGP -lr 0.01 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================

# =================================================================== fmnist ======================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedSTAR -lr 0.05 -alr 0.01 -sas 10 -sac 1.0 -dr 0.05 -uf True -ut True -udg True -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedALA -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedMTL -lr 0.01 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedProto -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedRep -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedPAC -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo Ditto -lr 0.01 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedTGP -lr 0.02 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================

# =================================================================== Office-31 ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedSTAR -lr 0.05 -alr 0.01 -sas 10 -ut True -uf True -udg True -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo Ditto -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedALA -lr 0.4 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedMTL -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedProto -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedRep -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedPAC -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedTGP -lr 0.01 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================
done
