import copy
import torch
import time
from flcore.clients.clientala import *
from flcore.servers.serverbase import Server
from concurrent.futures import ThreadPoolExecutor, as_completed

class FedALA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        torch.set_num_threads(1)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(i)

            # for client in self.selected_clients:
            #     client.train()
            
            max_workers = min(self.max_parallel_clients, len(self.selected_clients))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(c.train) for c in self.selected_clients]
                # 예외 전파 및 완료 대기
                for f in as_completed(futures):
                    f.result()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)
