import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA
from sklearn.preprocessing import label_binarize

class clientALA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        
    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

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
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)


    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0.0

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # base/head 구조 지원 + fallback
                if hasattr(self.model, "base") and hasattr(self.model, "head"):
                    rep = self.model.base(x)
                    output = self.model.head(rep)
                else:
                    output = self.model(x)

                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (output.argmax(dim=1) == y).sum().detach()

        train_correct = train_correct.item() if isinstance(train_correct, torch.Tensor) else train_correct
        return losses, train_num, train_correct


    def test_metrics(self, test_loader=None):
        # 1) 테스트 로더 준비
        testloader = self.load_test_data() if test_loader is None else test_loader
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        # 2) 결과 저장용
        y_prob, y_true = [], []
        test_correct = 0
        test_num = 0

        # 3) 테스트 루프
        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # base/head 구조 지원 + fallback
                if hasattr(self.model, "base") and hasattr(self.model, "head"):
                    rep = self.model.base(x)
                    output = self.model.head(rep)
                else:
                    output = self.model(x)

                probs = F.softmax(output, dim=1).detach().cpu().numpy()

                # one-hot labels
                truths = label_binarize(
                    y.detach().cpu().numpy(),
                    classes=np.arange(self.num_classes)
                )

                test_correct += (output.argmax(dim=1) == y).sum().detach()
                test_num += y.size(0)

                y_prob.append(probs)
                y_true.append(truths)

        # 4) concat (비어있을 때 안전 처리)
        if y_prob:
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
        else:
            y_prob = np.zeros((0, self.num_classes))
            y_true = np.zeros((0, self.num_classes))

        test_correct = test_correct.item() if isinstance(test_correct, torch.Tensor) else test_correct
        return test_correct, test_num, y_prob, y_true