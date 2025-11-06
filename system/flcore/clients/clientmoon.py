import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.global_model = None
        self.old_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                rep_old = self.old_model.base(x).detach()
                rep_global = self.global_model.base(x).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        self.old_model = copy.deepcopy(self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model = model

    def train_metrics(self):
        trainloader = self.load_train_data()

        # 모델들을 장치로 이동 & eval
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        # old/global 모델도 같은 장치에서 사용 (forward만, grad X)
        if hasattr(self, "old_model") and self.old_model is not None:
            self.old_model.to(self.device, non_blocking=True)
            self.old_model.eval()
        if hasattr(self, "global_model") and self.global_model is not None:
            self.global_model.to(self.device, non_blocking=True)
            self.global_model.eval()

        train_num = 0
        losses = 0.0
        train_correct = 0

        with torch.no_grad():
            for x, y in trainloader:
                # 입력 정리
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # 현재 모델 forward
                rep = self.model.base(x)            # [B, D]
                output = self.model.head(rep)       # [B, C]
                cls_loss = self.loss(output, y)

                # MOON contrastive term
                # a = sim(rep, global)/tau, b = sim(rep, old)/tau
                # -log( exp(a)/(exp(a)+exp(b)) ) = softplus(b - a)
                if hasattr(self, "old_model") and self.old_model is not None \
                and hasattr(self, "global_model") and self.global_model is not None:
                    rep_old    = self.old_model.base(x).detach()      # [B, D]
                    rep_global = self.global_model.base(x).detach()   # [B, D]

                    a = F.cosine_similarity(rep, rep_global, dim=1) / self.tau  # [B]
                    b = F.cosine_similarity(rep, rep_old,    dim=1) / self.tau  # [B]
                    loss_con = F.softplus(b - a).mean()  # 수치 안정화된 대비 손실
                    loss = cls_loss + self.mu * loss_con
                else:
                    loss = cls_loss

                # 집계
                batch_size = y.size(0)
                train_num += batch_size
                losses    += loss.item() * batch_size

                preds = torch.argmax(output, dim=1)
                train_correct += (preds == y).sum().item()

        return losses, train_num, train_correct