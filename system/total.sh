#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 각 명령의 -did 한 곳만 수정하면 같은 값을 Python 실행 전 GPU mask에도 적용한다.
if [[ -n "${PYTHON_BIN:-}" ]]; then
    training_python="$PYTHON_BIN"
elif [[ -x "$HOME/miniconda3/envs/mincheol/bin/python" ]]; then
    training_python="$HOME/miniconda3/envs/mincheol/bin/python"
else
    training_python="python"
fi

run_experiment() {
    local selected_gpu="${CUDA_VISIBLE_DEVICES-0}"
    local -a experiment_args=("$@")
    local gpu_arg_seen=0
    local index
    for ((index=0; index<${#experiment_args[@]}; index++)); do
        case "${experiment_args[index]}" in
            -did|--device_id|-did=*|--device_id=*)
                if (( gpu_arg_seen )); then
                    echo "GPU 인자가 중복되었습니다. -did 또는 --device_id는 한 번만 지정하세요." >&2
                    return 2
                fi
                gpu_arg_seen=1
                ;;
        esac
        case "${experiment_args[index]}" in
            -did|--device_id)
                if (( index + 1 >= ${#experiment_args[@]} )); then
                    echo "-did 뒤에 GPU 번호를 지정해야 합니다." >&2
                    return 2
                fi
                selected_gpu="${experiment_args[index+1]}"
                ;;
            -did=*|--device_id=*)
                selected_gpu="${experiment_args[index]#*=}"
                ;;
        esac
    done
    echo "[Launch] GPU=${selected_gpu} model command: ${training_python} -u main.py ${experiment_args[*]}"
    nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$selected_gpu" \
        "$training_python" -u main.py "${experiment_args[@]}" < /dev/null
}

# 각 실험(-t 반복 포함)이 끝나면 결과 저장 후 그 실험의 checkpoint만 자동 삭제한다.
# 결과/로그/데이터/사전학습 모델은 유지하며, 실행마다 별도 폴더로 다음 실험과 격리한다.
# 여섯 FedRG 구성을 GPU별로 실행: bash total.sh fedrg-all
if [ "${1:-}" = "fedrg-all" ]; then
    shift
    exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_fedrg_all.sh" "$@"
fi

model=HtFE-img-8
t=3
global_rounds=500
batch_size=32
join_ratio=1
num_clients=20
num_classes=102 # 100 102 200
local_epoch=1
dataset=Flowers102 # Cifar100 Flowers102 TinyImagenet
did=0

# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did ${did} -algo Local > total-${dataset}-${model}-fd=512-Local.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 1 -algo FedProto -lam 10 > total-${dataset}-${model}-fd=512-FedProto.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 1 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 100 > total-${dataset}-${model}-fd=512-FedGen.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 2 -algo FD -lam 1 > total-${dataset}-${model}-fd=512-FD.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 2 -algo FML -al 0.5 -bt 0.5 > total-${dataset}-${model}-fd=512-FML.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 3 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 > total-${dataset}-${model}-fd=512-FedKD.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 3 -algo LG-FedAvg > total-${dataset}-${model}-fd=512-LG-FedAvg.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 3 -algo FedGH -slr 0.01 -se 1 > total-${dataset}-${model}-fd=512-FedGH.out 2>&1 &
# run_experiment -did ${did} -t ${t} -gr ${global_rounds} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs ${batch_size} -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -algo FedTGP -lam 10 -se 100 -mart 500 --max_parallel_clients 1 -eg 10 > total-${dataset}-${model}-fd=512-FedTGP_did_${did}.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 5 -algo FedKTL-stylegan-xl -slr 0.01 -sbs 100 -se 100 -lam 1 -mu 50 > total-${dataset}-${model}-fd=512-FedKTL-stylegan-xl.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 2 -algo FedKTL-stylegan-3 -slr 0.01 -sbs 100 -se 100 -lam 0.1 -mu 50 -GPath stylegan/stylegan-3-models/Benches-512.pkl > total-${dataset}-${model}-fd=512-FedKTL-stylegan-3.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 1 -algo FedKTL-stable-diffusion -slr 0.1 -sbs 100 -se 100 -lam 0.01 -mu 100 -GPath stable-diffusion/v1.5 > total-${dataset}-${model}-fd=512-FedKTL-stable-diffusion.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs 10 -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -did 4 -algo FedMRL -sfd 128 > total-${dataset}-${model}-fd=512-FedMRL.out 2>&1 &
# run_experiment -t ${t} -ab 1 -lr 0.01 -jr ${join_ratio} -ls ${local_epoch} -lbs ${batch_size} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -algo FedSA -gr ${global_rounds} -did ${did} -lam 10 -mcl 1.0 -cc 1.0 --max_parallel_clients 1 --data_loader_num_workers 0 --pin_memory True > "total-${dataset}-${model}-fd=512-FedSA_lam10.0_mcl1.0_cc1.0.out" 2>&1 &

# 이종 모델 비교 대상: FedProtoKD / FedRG
# FedHPro는 동종 모델 가중치 집계가 필요하므로 이번 비교 실험에서 제외한다.
# 원하는 명령의 맨 앞 #을 제거해 실행한다. 
# 전체 옵션/공식 구현과의 차이: INTEGRATION.md

# FedProtoKD: 기존 데이터는 보존하고, train에서 public 50장/class를 분리한 전용 split 사용.
# 준비 명령 (새 데이터셋을 사용할 때 한 번 실행):
# python dataset/prepare_protokd_public.py --source "dataset/${dataset}" --output "dataset/${dataset}_ProtoKD" --samples-per-class 50 --seed 42
protokd_dataset="${PROTOKD_DATASET:-${dataset}_ProtoKD}"
# 외부 공개 데이터를 쓰려면 PROTOKD_DATASET과 PROTOKD_PUBLIC_PATH를 명시한다.
# run_experiment -t ${t} -gr ${global_rounds} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs ${batch_size} -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data "${protokd_dataset}" -m ${model} -fd 512 -did ${did} -algo FedProtoKD --protokd_public_path "${PROTOKD_PUBLIC_PATH:-dataset/${protokd_dataset}/public.npz}" --protokd_global_epochs 10 --protokd_client_kd_epochs 5 --protokd_tgp_epochs 100 --protokd_batch_size 16 --protokd_temperature 2.0 --protokd_epsilon 0.4 --protokd_gamma 0.5 --protokd_theta 0.5 --protokd_margin 50 --max_parallel_clients 1 -eg 10 > "total-${protokd_dataset}-${model}-fd=512-FedProtoKD.out" 2>&1 &

# FedRG: uniform은 공식 코드의 graph target 덮어쓰기를 유지한다.
# 신뢰도 가중치로 학습하는 변형은 --rg_graph_target reliability로 명시한다.
run_experiment -did ${did} -t ${t} -gr ${global_rounds} -ab 1 -lr 0.01 -jr ${join_ratio} -lbs ${batch_size} -ls ${local_epoch} -nc ${num_clients} -ncl ${num_classes} -data ${dataset} -m ${model} -fd 512 -algo FedRG -slr 0.01 --rg_graph_target uniform --node_dim 40 --adjalpha 3 --gc_epoch 10 --gnn_learning_rate 0.001 --gnn_weight_decay 0.0001 --max_parallel_clients 1 -eg 10 > "total-${dataset}-${model}-fd=512-FedRG.out" 2>&1 &
