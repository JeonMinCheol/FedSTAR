import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client


class clientTGP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        # in-memory buffers
        self.global_protos = None
        self.protos = None

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos) if global_protos is not None else None

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            surrounding_global = self.global_protos  
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if surrounding_global is not None:
                    proto_target = rep.detach().clone()
                    for idx, yy in enumerate(y):
                        c = int(yy.item())
                        gp = surrounding_global.get(c, None) if isinstance(surrounding_global, dict) else None
                        if gp is not None and not isinstance(gp, list):
                            proto_target[idx, :] = gp.data
                    loss = loss + self.loss_mse(proto_target, rep) * self.lamda

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        self.collect_protos()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                for idx, yy in enumerate(y):
                    c = int(yy.item())
                    protos[c].append(rep[idx, :].detach())

        self.protos = agg_func(protos) 

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0.0

        surrounding_global = self.global_protos
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if surrounding_global is not None:
                    proto_target = rep.detach().clone()
                    for idx, yy in enumerate(y):
                        c = int(yy.item())
                        gp = surrounding_global.get(c, None) if isinstance(surrounding_global, dict) else None
                        if gp is not None and not isinstance(gp, list):
                            proto_target[idx, :] = gp.data
                    loss = loss + self.loss_mse(proto_target, rep) * self.lamda

                bs = y.size(0)
                train_num += bs
                losses += loss.item() * bs
                train_correct += (output.argmax(dim=1) == y).sum().item()

        return losses, train_num, train_correct


def agg_func(protos):
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = torch.zeros_like(proto_list[0])
            for p in proto_list:
                proto += p
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos
