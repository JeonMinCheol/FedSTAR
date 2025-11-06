import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from utils.data_utils import read_client_data

import matplotlib.pyplot as plt

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.num_workers = args.num_workers
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.dirchlet = args.dirchlet
        self.pt_path = f"../dataset/models/{self.dataset}/{self.algorithm}"
        self.plt_path = f"./embedding/{self.dataset}/{self.algorithm}/{self.dirchlet}/"

        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        
        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=0)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=0)
    
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, str(self.id) + "_" + item_name + ".pt"))

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                rep = self.model.base(x)        
                output = self.model.head(rep)   
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).detach()
        train_correct = train_correct.item() if isinstance(train_correct, torch.Tensor) else train_correct

        return losses, train_num, train_correct

    def test_metrics(self, test_loader=None):
        # 1) 테스트 로더 준비
        testloader = self.load_test_data()
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        # 2) 결과 저장용
        y_prob, y_true = [], []
        test_correct = 0
        test_num     = 0

        with torch.no_grad():
            for x, y in testloader:
                # (a) x, y를 올바른 텐서로
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                rep = self.model.base(x)        
                output = self.model.head(rep)      # [b, C]
                probs  = F.softmax(output, dim=1).cpu().numpy()  # [b, C]
                truths = label_binarize(
                    y.cpu().numpy(),
                    classes=np.arange(self.num_classes)
                )                                              # [b, C]

                # (d) accuracy 집계
                test_correct += (output.argmax(dim=1) == y).sum().detach()
                test_num     += y.size(0)

                y_prob.append(probs)
                y_true.append(truths)

        # 5) 최종 concatenate
        if y_prob:
            y_prob = np.concatenate(y_prob, axis=0)  # (N_test, C)
            y_true = np.concatenate(y_true, axis=0)
        else:
            # 전혀 평가할 샘플 없으면 빈 배열
            y_prob = np.zeros((0, self.num_classes))
            y_true = np.zeros((0, self.num_classes))

        test_correct = test_correct.item() if isinstance(test_correct, torch.Tensor) else test_correct
        return test_correct, test_num, y_prob, y_true

def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))

def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"))
    except FileNotFoundError:
        return None