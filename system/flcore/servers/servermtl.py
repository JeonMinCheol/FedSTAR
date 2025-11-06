import time
import torch
from flcore.clients.clientmtl import clientMTL
from flcore.servers.serverbase import Server
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os, time
from utils.visual import *

class FedMTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.dim = len(self.flatten(self.global_model))
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=args.device)
        self.device = args.device

        I = torch.ones((self.num_join_clients, self.num_join_clients))
        i = torch.ones((self.num_join_clients, 1))
        omega = (I - 1 / self.num_join_clients * i.mm(i.T)) ** 2
        self.omega = omega.to(args.device)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMTL)
        self.Budget = []
            
        print(f"\nJoin clients / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # CPU oversubscription 방지
        torch.set_num_threads(1)
        self.max_parallel_clients = getattr(args, "parallel_clients", min(8, os.cpu_count() or 8))

        print(f"\nJoin clients / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # (1) 서버 파라미터 집계
            self.aggregate_parameters()

            print(f"\n------------- Round number: {i} -------------")
            if i % self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(i)

            # (2) 각 클라이언트에 파라미터 전송
            for idx, client in enumerate(self.selected_clients):
                start_time = time.time()
                client.set_parameters(self.W_glob, self.omega, idx)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            # =====================================================
            # (3) 병렬 클라이언트 학습
            # =====================================================

            # for client in self.selected_clients:
            #     client.train()
            
            print(f"[Server] Training {len(self.selected_clients)} clients (parallel={self.max_parallel_clients})...")
            torch.cuda.empty_cache()
            with ThreadPoolExecutor(max_workers=self.max_parallel_clients) as executor:
                futures = [executor.submit(c.train) for c in self.selected_clients]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"[Warning] Client thread failed: {e}")

            # =====================================================
            # (4) 라운드 종료 후 정리
            # =====================================================
            torch.cuda.empty_cache()
            self.Budget.append(time.time() - s_t)
            print('-' * 50, f"Round time: {self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def flatten(self, model):
        state_dict = model.state_dict()
        keys = state_dict.keys()
        W = [state_dict[key].flatten() for key in keys]
        return torch.cat(W)

    def aggregate_parameters(self):
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=self.device)
        for idx, client in enumerate(self.selected_clients):
            self.W_glob[:, idx] = self.flatten(client.model)
