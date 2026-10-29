import copy
import torch
import torch.nn as nn
import numpy as np
import os
from weakref import WeakValueDictionary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.role = 'Client_' + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        self.data_loader_num_workers = int(getattr(args, "data_loader_num_workers", 0))
        self.pin_memory = bool(getattr(args, "pin_memory", self.device == "cuda"))

        if args.models_folder_name:
            model = load_item(self.role, 'model', args.models_folder_name).to(self.device)
            print('load pre-trained model from', args.models_folder_name)
        elif args.algorithm == "FedDSA":
            from flcore.trainmodel.feddsa_model import build_feddsa_model_from_spec
            model_spec = args.models[self.id % len(args.models)]
            model = build_feddsa_model_from_spec(model_spec, args).to(self.device)
        elif args.algorithm == "FedProtoKD":
            from flcore.trainmodel.prototype import build_protokd_model
            model_spec = args.models[self.id % len(args.models)]
            model = build_protokd_model(model_spec, args).to(self.device)
        else:
            model = BaseHeadSplit(args, self.id).to(self.device)
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return self.create_data_loader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return self.create_data_loader(test_data, batch_size, drop_last=False, shuffle=False)

    def create_data_loader(self, dataset, batch_size, drop_last=False, shuffle=False):
        kwargs = {
            "batch_size": batch_size,
            "drop_last": drop_last,
            "shuffle": shuffle,
            "num_workers": self.data_loader_num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.data_loader_num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **kwargs)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


_model_stores = WeakValueDictionary()


class ModelStore:
    """Model snapshots owned by one server run, without checkpoint I/O."""

    def __init__(self, path, device="cpu"):
        self.path = os.path.abspath(path)
        self.device = torch.device(device)
        self.models = {}
        _model_stores[self.path] = self

    def flush(self):
        os.makedirs(self.path, exist_ok=True)
        for (role, name), model in self.models.items():
            torch.save(copy_model(model, "cpu"), os.path.join(self.path, f"{role}_{name}.pt"))


def copy_model(model, device):
    # Copy directly to the destination, without retaining gradients or a temporary CUDA model.
    memo = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        value = tensor.detach().to(device, copy=True).requires_grad_(tensor.requires_grad)
        if isinstance(tensor, nn.Parameter):
            value = nn.Parameter(value, requires_grad=tensor.requires_grad)
        memo[id(tensor)] = value
    return copy.deepcopy(model, memo).to(device)


def item_exists(role, item_name, item_path):
    store = _model_stores.get(os.path.abspath(item_path))
    return (store is not None and (role, item_name) in store.models) or os.path.isfile(
        os.path.join(item_path, f"{role}_{item_name}.pt")
    )


def save_item(item, role, item_name, item_path=None):
    store = _model_stores.get(os.path.abspath(item_path))
    if store is not None and isinstance(item, nn.Module):
        store.models[role, item_name] = copy_model(item, store.device)
        return
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))

def move_loaded_item(item, device):
    if isinstance(item, nn.Module):
        return item.to(device)
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return {key: move_loaded_item(value, device) for key, value in item.items()}
    if isinstance(item, list):
        return [move_loaded_item(value, device) for value in item]
    if isinstance(item, tuple):
        return tuple(move_loaded_item(value, device) for value in item)
    return item

def load_item(role, item_name, item_path=None):
    try:
        store = _model_stores.get(os.path.abspath(item_path))
        if store is not None and (role, item_name) in store.models:
            # Callers train and zero aggregation targets in place; never share snapshots.
            item = copy.deepcopy(store.models[role, item_name])
            if store.device.type == "cuda":
                return item
        else:
            path = os.path.join(item_path, role + "_" + item_name + ".pt")
            try:
                item = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                item = torch.load(path, map_location="cpu")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return move_loaded_item(item, "cuda")
        return item
    except FileNotFoundError:
        print(role, item_name, 'Not Found')
        return None
