from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch, os, time
from utils.visual import *

class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # CPU oversubscription 방지
        torch.set_num_threads(1)

        # 병렬 클라이언트 개수 (기본: 코어 수 또는 args.parallel_clients)
        self.max_parallel_clients = 8


    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n------------- Round number: {i} -------------")
                print("\nEvaluate global model")
                self.evaluate(i)

            # =====================================================
            # (1) 병렬 클라이언트 학습
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
                        print(f"[Warning] client thread failed: {e}")

            # =====================================================
            # (2) 모델 수집 및 서버 집계
            # =====================================================
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # =====================================================
            # (3) 라운드 타이밍 및 로그
            # =====================================================
            self.Budget.append(time.time() - s_t)
            print('-' * 50, f"Round time: {self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # =====================================================
        # (4) 학습 종료 후 결과 요약
        # =====================================================
        print("\n⏱️  Avg time per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientMOON)
        #     print(f"\n------------- Fine tuning round -------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate(i)