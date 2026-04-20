#!/usr/bin/env python
import copy
import torch
import argparse
import os
import random
import time
import warnings
import numpy as np
import logging

from flcore.servers.servermtl import FedMTL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverproto import FedProto
from flcore.servers.serverala import FedALA
from flcore.servers.serverstar import FedSTAR
from flcore.servers.serverpac import FedPAC
from flcore.servers.servertgp import FedTGP

from flcore.trainmodel.models import *
from flcore.trainmodel.fedstar_model import build_fedstar_model, parse_client_model_names

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.mobilenet_v3 import *
from flcore.trainmodel.transformer import *

from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def set_random_seed(seed: int, deterministic: bool = True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    fedstar_client_backbones = None
    fedstar_bootstrap_model = model_str

    if args.algorithm == "FedSTAR":
        fedstar_client_backbones = parse_client_model_names(
            client_models=args.client_models,
            num_clients=args.num_clients,
            default_model=model_str,
        )
        args.client_backbones = fedstar_client_backbones
        fedstar_bootstrap_model = fedstar_client_backbones[0]

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        active_model_str = fedstar_bootstrap_model if args.algorithm == "FedSTAR" else model_str
        args.model_factory = None

        # Generate args.model
        if active_model_str == "mlr": # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)
                
        elif active_model_str == "fmnist": # non-convex
            args.model = FashionCNNModel().to(args.device)

        elif active_model_str == "cnn": # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Tiny-imagenet" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            elif "office" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=379456).to(args.device)

        elif active_model_str == "dnn": # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif active_model_str == "resnet":
            import torchvision
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif active_model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif active_model_str == "googlenet":
            import torchvision
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif active_model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        elif active_model_str == "mobilenet_v3":
            args.model = mobilenet_v3_ultralite(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif active_model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif active_model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=emb_dim).to(args.device)

        elif active_model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif active_model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)

        elif active_model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2, 
                            num_classes=args.num_classes).to(args.device)
        
        elif active_model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif active_model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        # print(args.model)

        # select algorithm
        if args.algorithm == "FedSTAR":
            def fedstar_model_factory(
                client_id,
                backbones=tuple(fedstar_client_backbones),
                dataset=args.dataset,
                num_classes=args.num_classes,
                shared_dim=args.shared_dim,
                private_dim=args.private_dim,
                use_private_branch=args.use_private_branch,
                shared_classifier_scale=args.shared_classifier_scale,
            ):
                backbone_name = backbones[int(client_id) % len(backbones)]
                return build_fedstar_model(
                    model_name=backbone_name,
                    dataset=dataset,
                    num_classes=num_classes,
                    shared_dim=shared_dim,
                    private_dim=private_dim,
                    use_private_branch=use_private_branch,
                    shared_classifier_scale=shared_classifier_scale,
                )

            args.model_factory = fedstar_model_factory
            args.model = fedstar_model_factory(0)
            server = FedSTAR(args, i)

        elif args.algorithm == "FedMTL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedMTL(args, i)

        elif args.algorithm == "FedTGP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedTGP(args, i)

        elif args.algorithm == "Ditto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedALA":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-seed', "--random_seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument('-dcl', "--dirchlet", type=float, default=1.0, 
                        help="dirchlet value")
    parser.add_argument('-nw', "--num_workers", type=int, default=0)
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="1")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_epochs", type=int, default=3, 
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
    parser.add_argument('-eg', "--eval_gap", type=int, default=5,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=128)

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
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=10)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=5)
    parser.add_argument('-dmu', "--ditto_mu", type=int, default=1)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=-1,
                        help="More fine-graind than its original paper.")
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    # FedSTAR
    parser.add_argument('-dr', "--dropout", type=float, default=0.05)
    parser.add_argument('-uf', "--use_film", type=str2bool, default=False)
    parser.add_argument('-ut', "--use_transformer", type=str2bool, default=False)
    parser.add_argument('-udg', "--use_decompose_with_global", type=str2bool, default=False)
    parser.add_argument('-sas', "--server_agg_steps", type=int, default=5)
    parser.add_argument('-sac', "--server_agg_clip", type=float, default=1.0)
    parser.add_argument('-alr', "--aggregator_learning_rate", type=float, default=0.005)
    parser.add_argument("--client_models", type=str, default="")
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=128)
    parser.add_argument("--lambda_align", type=float, default=0.3)
    parser.add_argument("--lambda_sep", type=float, default=1.0)
    parser.add_argument("--anchor_beta_ema", type=float, default=0.9)
    parser.add_argument("--use_private_branch", type=str2bool, default=True)
    parser.add_argument("--use_separation_loss", type=str2bool, default=True)
    parser.add_argument("--use_ema", type=str2bool, default=True)
    parser.add_argument("--normalize_shared_align", type=str2bool, default=True)
    parser.add_argument("--use_anchor_softmax", type=str2bool, default=True)
    parser.add_argument("--anchor_softmax_weight", type=float, default=1.0)
    parser.add_argument("--anchor_center_weight", type=float, default=0.5)
    parser.add_argument("--anchor_cosface_margin", type=float, default=0.15)
    parser.add_argument("--anchor_cosface_scale", type=float, default=16.0)
    parser.add_argument("--shared_classifier_scale", type=float, default=16.0)
    parser.add_argument("--use_amp", type=str2bool, default=True)

    args = parser.parse_args()
    set_random_seed(args.random_seed, deterministic=args.deterministic)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Random seed: {}".format(args.random_seed))
    print("Deterministic: {}".format(args.deterministic))
    print("Dataset: {}".format(args.dataset))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    # print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    # print("Clients randomly join: {}".format(args.random_join_ratio))
    # print("Client drop rate: {}".format(args.client_drop_rate))
    # print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    # print("Running times: {}".format(args.times))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    # print("Auto break: {}".format(args.auto_break))

    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    # print("DLG attack: {}".format(args.dlg_eval))
    
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    # print("Total number of new clients: {}".format(args.num_new_clients))
    # print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    # print("Dirchlet rate: {}".format(args.dirchlet))
    
    if args.algorithm == "FedSTAR":
        client_backbones = parse_client_model_names(
            client_models=args.client_models,
            num_clients=args.num_clients,
            default_model=args.model,
        )
        # print("client_backbones: {}".format(client_backbones))
        print("shared_dim/private_dim: {}/{}".format(args.shared_dim, args.private_dim))
        print("lambda_align/lambda_sep: {}/{}".format(args.lambda_align, args.lambda_sep))
        print("anchor_beta_ema: {}".format(args.anchor_beta_ema))
        print("use_private_branch: {}".format(args.use_private_branch))
        print("use_separation_loss: {}".format(args.use_separation_loss))
        print("use_ema: {}".format(args.use_ema))
        print("normalize_shared_align: {}".format(args.normalize_shared_align))
        print("use_anchor_softmax: {}".format(args.use_anchor_softmax))
        print("anchor_cosface_margin/scale/weight: {}/{}/{}".format(
            args.anchor_cosface_margin,
            args.anchor_cosface_scale,
            args.anchor_softmax_weight,
        ))
        print("shared_classifier_scale: {}".format(args.shared_classifier_scale))
        print("anchor_center_weight: {}".format(args.anchor_center_weight))
       
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("=" * 50)

    run(args)
