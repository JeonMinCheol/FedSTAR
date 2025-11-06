import copy
import numpy as np
import time
import torch
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time
from sklearn import metrics
from utils.visual import *

class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # CPU oversubscription 방지
        torch.set_num_threads(1)
        # 동시 병렬 클라이언트 수 (코어 수/환경에 맞게 조정)
        self.max_parallel_clients = 8

    def _run_client_round(self, client):
        # Ditto는 personalization step(ptrain) 후 local train을 이어서 수행
        client.ptrain()
        client.train()

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {i}-------------")

            if i % self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(i)

            # ================== 병렬 클라이언트 학습 ==================

            # for client in self.selected_clients:
            #     self._run_client_round(client)

            torch.cuda.empty_cache()
            max_workers = min(self.max_parallel_clients, len(self.selected_clients))
            print(f"[Server] Training {len(self.selected_clients)} clients (parallel={max_workers})...")

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self._run_client_round, c) for c in self.selected_clients]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"[Warning] client thread failed: {e}")

            # ================== 모델 수신 및 집계 ==================
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # ================== 라운드 종료 ==================
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, f"{self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientDitto)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate(i)