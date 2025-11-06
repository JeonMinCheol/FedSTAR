from sklearn.preprocessing import label_binarize
import torch
import numpy as np
import time
import copy
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F

class clientDitto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.ditto_mu
        self.plocal_steps = args.plocal_steps

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )


    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device, non_blocking=True)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
    def ptrain(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.to(self.device, non_blocking=True)
        self.model_per.train()

        max_local_epochs = self.plocal_steps
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model_per(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model.parameters(), self.device)

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        trainloader = self.load_train_data()

        # 기본/개인화 모델 장치/모드
        self.model.to(self.device, non_blocking=True)
        self.model_per.to(self.device, non_blocking=True)
        self.model_per.eval()

        train_num = 0
        losses    = 0.0
        train_correct = 0

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # forward
                output = self.model_per(x)

                # 분류 손실
                loss = self.loss(output, y)

                # Ditto proximal term: || theta - theta_personal ||
                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss = loss + 0.5 * self.mu * torch.norm(gm - pm, p=2)

                # 집계
                batch_size = y.size(0)
                train_num += batch_size
                losses    += loss.item() * batch_size

                preds = torch.argmax(output, dim=1)
                train_correct += (preds == y).sum().item()

        return losses, train_num, train_correct


    def test_metrics(self, xx):
        # 모델 장치/모드
        testloader = self.load_test_data()
        self.model_per.to(self.device, non_blocking=True)
        self.model_per.eval()

        test_correct = 0
        test_num     = 0
        y_prob_list, y_true_list = [], []

        with torch.no_grad():
            for x, y in testloader:
                # 입력 정리
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # forward
                output = self.model_per(x)             # [B, C]
                probs  = F.softmax(output, dim=1)      # [B, C]
                preds  = torch.argmax(probs, dim=1)    # [B]

                # 집계
                test_correct += (preds == y).sum().item()
                test_num     += y.size(0)

                y_prob_list.append(probs.detach().cpu().numpy())
                y_true_list.append(
                    label_binarize(
                        y.detach().cpu().numpy(),
                        classes=np.arange(self.num_classes)
                    )
                )

        # 컨캐튼 (비었을 때 (0, C))
        if y_prob_list:
            y_prob = np.concatenate(y_prob_list, axis=0)
            y_true = np.concatenate(y_true_list, axis=0)
        else:
            y_prob = np.zeros((0, self.num_classes), dtype=float)
            y_true = np.zeros((0, self.num_classes), dtype=int)

        return test_correct, test_num, y_prob, y_true