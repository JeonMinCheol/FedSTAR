#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/test_${RUN_TAG}.txt}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[test.sh] Logging to: $LOG_FILE"
echo "[test.sh] Started at: $(date '+%Y-%m-%d %H:%M:%S')"

model=mobilenet_v3
dcl=0.1
lbs=256
jr=0.3
ls=3
nc=100
gr=80
seeds=(42)

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "[test.sh] Fixed seeds: ${seeds[*]}"
echo "[test.sh] CUBLAS_WORKSPACE_CONFIG: $CUBLAS_WORKSPACE_CONFIG"

for seed in "${seeds[@]}"; do
export PYTHONHASHSEED="$seed"
echo "[test.sh] Running seed: $seed"
echo "[test.sh] PYTHONHASHSEED: $PYTHONHASHSEED"
# =================================================================== CIFAR-100 ===================================================================
python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedSTAR -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo Ditto -lr 0.15 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedALA -lr 0.5 -et 1.0 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedRep -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedMTL -lr 0.01 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedProto -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedPAC -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 100 -ls $ls -data Cifar100 -m $model -algo FedTGP -lr 0.01 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================

# =================================================================== DomainNet ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedSTAR -lr 0.015 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedALA -lr 0.2 -et 1.0 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedPAC -lr 0.15 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedMTL -lr 0.005 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedProto -lr 0.005 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo Ditto -lr 0.005 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedRep -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 345 -ls $ls -data DomainNet -m $model -algo FedTGP -lr 0.01 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================

# =================================================================== fmnist ======================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedSTAR -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedALA -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedMTL -lr 0.01 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedProto -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedRep -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedPAC -lr 0.1 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo Ditto -lr 0.01 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 10 -ls $ls -data fmnist -m $model -algo FedTGP -lr 0.02 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================

# =================================================================== Office-31 ===================================================================
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedSTAR -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo Ditto -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedALA -lr 0.4 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedMTL -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedProto -lr 0.02 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedRep -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedPAC -lr 0.05 -seed $seed
# python -u main.py -gr $gr -dcl $dcl -lbs $lbs -nc $nc -jr $jr -nb 31 -ls $ls -data Office-31 -m $model -algo FedTGP -lr 0.01 -lam 0.1 -se 100 -mart 100 -seed $seed
# =================================================================================================================================================
done
