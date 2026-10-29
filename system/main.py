#!/usr/bin/env python
import argparse
import os
from utils.device_utils import bootstrap_cuda_environment, configure_runtime_device, gpu_ids

# This must run before importing torch or any flcore/torchvision modules.
_device_selection = bootstrap_cuda_environment() if __name__ == "__main__" else None

import torch
import time
import warnings
import numpy as np
import logging
import random

from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from flcore.servers.servergen import FedGen
from flcore.servers.serverfd import FD
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.servergh import FedGH
from flcore.servers.servertgp import FedTGP
from flcore.servers.serverdsa import FedDSA
from flcore.servers.serverktl_stylegan_xl import FedKTL as FedKTL_stylegan_xl
from flcore.servers.serverktl_stylegan_3 import FedKTL as FedKTL_stylegan_3
from flcore.servers.serverktl_stable_diffusion import FedKTL as FedKTL_stable_diffusion
from flcore.servers.servermrl import FedMRL
from flcore.servers.serversa import FedSA
from flcore.servers.serverhpro import FedHPro
from flcore.servers.serverprotokd import FedProtoKD
from flcore.servers.serverfedrg import FedRG

try:
    from flcore.servers.serverl2g_feat import FedL2G as FedL2G_feat
    from flcore.servers.serverl2g_logit import FedL2G as FedL2G_logit
except ModuleNotFoundError as e:
    FedL2G_feat = None
    FedL2G_logit = None
    FedL2G_import_error = e
else:
    FedL2G_import_error = None

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def configure_models(args):
    # Generate args.models
    if args.model_specs:
        args.models = args.model_specs

    elif args.model_family == "HPro-ResNet10":
        if args.feature_dim != 256:
            raise ValueError("HPro-ResNet10 uses the official 256-dimensional backbone; pass -fd 256")
        args.models = ['hpro_resnet10(nclasses=args.num_classes)']

    elif args.model_family == "Homo-img-CNN":
        args.models = ['FedAvgCNN(in_features=3, num_classes=args.num_classes)']

    elif args.model_family == "Homo-img-ResNet18":
        args.models = ['torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)']

    elif args.model_family == "HtFE-img-2":
        args.models = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
        ]

    elif args.model_family == "HtFE-img-3":
        args.models = [
            'resnet10(num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
        ]

    elif args.model_family == "HtFE-img-4":
        args.models = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
        ]

    elif args.model_family == "HtFE-img-5":
        args.models = [
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
        ]

    elif args.model_family == "HtFE-img-8":
        args.models = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
        ]

    elif args.model_family == "HtFE-img-9":
        args.models = [
            'resnet4(num_classes=args.num_classes)', 
            'resnet6(num_classes=args.num_classes)', 
            'resnet8(num_classes=args.num_classes)', 
            'resnet10(num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
        ]

    elif args.model_family == "HtFE-img-8-HtC-img-4":
        args.models = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
        ]
        args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes)'
        args.heads = [
            'Head(hidden_dims=[512], num_classes=args.num_classes)', 
            'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
            'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
            'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
        ]

    elif args.model_family == "Res34-HtC-img-4":
        args.models = [
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
        ]
        args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes)'
        args.heads = [
            'Head(hidden_dims=[512], num_classes=args.num_classes)', 
            'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
            'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
            'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
        ]

    elif args.model_family == "HtM-img-10":
        args.models = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
            'torchvision.models.vit_b_16(num_classes=args.num_classes)', 
            'torchvision.models.vit_b_32(num_classes=args.num_classes)'
        ]

    elif args.model_family == "HtFE-txt-2":
        args.models = [
            'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
        ]

    elif args.model_family == "HtFE-txt-4":
        args.models = [
            'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)'
        ]

    elif args.model_family == "HtFE-txt-5-1":
        args.models = [
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
        ]

    elif args.model_family == "HtFE-txt-5-2":
        args.models = [
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
        ]

    elif args.model_family == "HtFE-txt-5-3":
        args.models = [
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
        ]
        
    elif args.model_family == "HtFE-txt-6":
        args.models = [
            'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
            'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
            'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)', 
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
            'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'GRUNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
        ]

    elif args.model_family == "MLPs":
        args.models = [
            'AmazonMLP(feature_dim=[])', 
            'AmazonMLP(feature_dim=[500])', 
            'AmazonMLP(feature_dim=[1000, 500])', 
            'AmazonMLP(feature_dim=[1000, 500, 200])', 
        ]

    elif args.model_family == "MLP_1layer":
        args.models = [
            'AmazonMLP(feature_dim=[200])', 
            'AmazonMLP(feature_dim=[500])', 
        ]

    elif args.model_family == "MLP_layers":
        args.models = [
            'AmazonMLP(feature_dim=[500])', 
            'AmazonMLP(feature_dim=[1000, 500])', 
            'AmazonMLP(feature_dim=[1000, 500, 200])', 
        ]

    elif args.model_family == "HtFE-sen-2":
        args.models = [
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
        ]

    elif args.model_family == "HtFE-sen-3":
        args.models = [
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
        ]

    elif args.model_family == "HtFE-sen-5":
        args.models = [
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
            'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
            'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
        ]

    elif args.model_family == "HtFE-sen-8":
        args.models = [
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
            'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
            'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
            'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=2)',
            'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=3)',
            'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
            'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=2)',
        ]

    else:
        raise NotImplementedError
            

def run(args):

    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        configure_models(args)

        for model in args.models:
            print(model)

        if hasattr(args, 'global_model'):
            print('global_model:', args.global_model)

        if hasattr(args, 'heads'):
            for head in args.heads:
                print('head:', head)

        # select algorithm
        if args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedSA":
            server = FedSA(args, i)

        elif args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "FedHPro":
            server = FedHPro(args, i)

        elif args.algorithm == "FedProtoKD":
            server = FedProtoKD(args, i)

        elif args.algorithm == "FedRG":
            server = FedRG(args, i)

        elif args.algorithm == "FedGen":
            server = FedGen(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "LG-FedAvg":
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            server = FedKD(args, i)

        elif args.algorithm == "FedGH":
            server = FedGH(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)

        elif args.algorithm == "FedDSA":
            server = FedDSA(args, i)
            
        elif args.algorithm == "FedKTL-stylegan-xl":
            server = FedKTL_stylegan_xl(args, i)

        elif args.algorithm == "FedKTL-stylegan-3":
            server = FedKTL_stylegan_3(args, i)

        elif args.algorithm == "FedKTL-stable-diffusion":
            server = FedKTL_stable_diffusion(args, i)

        elif args.algorithm == "FedMRL":
            server = FedMRL(args, i)
            
        elif args.algorithm == "FedL2G-feat":
            if FedL2G_feat is None:
                raise FedL2G_import_error
            server = FedL2G_feat(args, i)

        elif args.algorithm == "FedL2G-logit":
            if FedL2G_logit is None:
                raise FedL2G_import_error
            server = FedL2G_logit(args, i)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('-am', "--anchor_momentum", type=float, default=0.9999)
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=gpu_ids, default=None,
                        help="GPU index/UUID or comma-separated IDs in PCI order. "
                             "If omitted, honor CUDA_VISIBLE_DEVICES; otherwise use GPU 0.")
    parser.add_argument('--check_device', action='store_true',
                        help="Print the selected GPU/CPU and exit without creating models or starting training")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="HtM10")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=10,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp',
                        help="Checkpoint root; each run uses a unique subdirectory")
    parser.add_argument('--model_storage_device', choices=['cpu', 'cuda'], default='cpu',
                        help="Keep model snapshots in CPU RAM (default) or training GPU VRAM")
    parser.add_argument('--keep_checkpoints', type=str2bool, nargs='?', const=True, default=False,
                        help="Write in-memory models to run checkpoints after saving results (default: discard them)")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-mfn', "--models_folder_name", type=str, default='',
                        help="The folder of pre-trained models")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--max_parallel_clients', type=int, default=1)
    parser.add_argument('--data_loader_num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=str2bool, nargs='?', const=True, default=True)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedProto
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    # FedHPro (official gradient matching and classifier calibration)
    parser.add_argument('--model_specs', nargs='+', default=None,
                        help="Optional explicit model expressions, overriding model_family")
    parser.add_argument('--length_of_hp', type=int, default=5)
    parser.add_argument('--rounds_of_hp', type=int, default=30)
    parser.add_argument('--hp_info_t', type=float, default=0.05)
    parser.add_argument('--lr_hp', type=float, default=0.1)
    parser.add_argument('--meta_length', type=int, default=50)
    parser.add_argument('--hpro_meta_rounds', type=int, default=100)
    parser.add_argument('--hpro_calibration_epochs', type=int, default=100)
    parser.add_argument('--batch_real', type=int, default=100)
    parser.add_argument('--hpro_gradient_repeats', type=int, default=1)
    parser.add_argument('--use_margin', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--dis_metric', choices=['matching', 'mse', 'cos'], default='matching')
    parser.add_argument('--hpro_dsa', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--dsa_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--hpro_local_augmentation', type=str2bool, nargs='?', const=True, default=True)
    # FedProtoKD: requires a public split separate from private train/test data
    parser.add_argument('--protokd_public_path', type=str, default='',
                        help="Public .npz file; default: dataset/<dataset>/public.npz")
    parser.add_argument('--protokd_global_model', type=str,
                        default='torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)')
    parser.add_argument('--protokd_learning_rate', type=float, default=None)
    parser.add_argument('--protokd_server_learning_rate', type=float, default=None)
    parser.add_argument('--protokd_weight_decay', type=float, default=None)
    parser.add_argument('--protokd_global_epochs', type=int, default=10)
    parser.add_argument('--protokd_client_kd_epochs', type=int, default=5)
    parser.add_argument('--protokd_tgp_epochs', type=int, default=100)
    parser.add_argument('--protokd_tgp_learning_rate', type=float, default=0.005)
    parser.add_argument('--protokd_batch_size', type=int, default=16)
    parser.add_argument('--protokd_temperature', type=float, default=2.0)
    parser.add_argument('--protokd_epsilon', type=float, default=0.4)
    parser.add_argument('--protokd_gamma', type=float, default=0.5)
    parser.add_argument('--protokd_theta', type=float, default=0.5)
    parser.add_argument('--protokd_margin', type=float, default=50.0)
    parser.add_argument('--protokd_classwise_margin', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--protokd_public_keep_ratio', type=float, default=1.0,
                        help="Official default keeps all public samples, sorted by prototype distance")
    parser.add_argument('--protokd_priority_ratio', type=float, default=0.8,
                        help="Fraction receiving increased CE weight after sorting by distance")
    parser.add_argument('--protokd_adjust_client_ce', type=str2bool, nargs='?', const=True, default=False)
    # FedRG: uniform reproduces the official aggregator's target overwrite
    parser.add_argument('--rg_graph_target', choices=['uniform', 'reliability'], default='uniform')
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--adjalpha', type=float, default=3.0)
    parser.add_argument('--gc_epoch', type=int, default=10)
    parser.add_argument('--gnn_weight_decay', type=float, default=0.0001)
    parser.add_argument('--gnn_learning_rate', type=float, default=0.001)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=100)
    # FML
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-bt', "--beta", type=float, default=1.0)
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.01)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    # FedDSA
    parser.add_argument('--shared_classifier_scale', type=float, default=5.0)
    parser.add_argument('--kappa', type=float, default=0.2)
    parser.add_argument('--use_amp', type=str2bool, nargs='?', const=True, default=False)
    # FedKTL
    parser.add_argument('-GPath', "--generator_path", type=str, default='stylegan/stylegan-xl-models/imagenet64.pkl')
    parser.add_argument('-prompt', "--stable_diffusion_prompt", type=str, default='a cat')
    parser.add_argument('-sbs', "--server_batch_size", type=int, default=100)
    parser.add_argument('-gbs', "--gen_batch_size", type=int, default=4,
                        help="Not related to the performance. A small value saves GPU memory.")
    parser.add_argument('-mu', "--mu", type=float, default=50.0)
    # FedMRL
    parser.add_argument('-sfd', "--sub_feature_dim", type=int, default=128)
    # FedL2G
    parser.add_argument('-mdb', "--meta_study_batches", type=int, default=1)
    parser.add_argument('-mqb', "--meta_quiz_batches", type=int, default=1)
    parser.add_argument('-wu', "--warm_up_rounds", type=int, default=50)
    # FedSA
    parser.add_argument('-mcl', '--mcl', "--margin_contrastive", dest="margin_contrastive", type=float, default=1.0)
    parser.add_argument('-cc', "--classifier_calibration", type=float, default=1.0)


    return parser


if __name__ == "__main__":
    total_start = time.time()
    args = build_parser().parse_args()
    configure_runtime_device(args, torch, _device_selection)
    if args.check_device:
        raise SystemExit(0)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)


    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
