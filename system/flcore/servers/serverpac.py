from flcore.clients.clientpac import clientPAC
from flcore.servers.serverbase import Server
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time
import numpy as np
import random
import torch
import cvxpy as cvx
import copy


class FedPAC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPAC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]

        self.Vars = []
        self.Hs = []
        self.uploaded_heads = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            self.Vars = []
            self.Hs = []
            for client in self.selected_clients:
                self.Vars.append(client.V)
                self.Hs.append(client.h)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate(i)

            print(f"[Server] Training {len(self.selected_clients)} clients (parallel={self.max_parallel_clients})...")
            # torch.cuda.empty_cache() # 잦은 호출은 오히려 성능 저하 원인일 수 있음 (필요시 주석 해제)
            
            with ThreadPoolExecutor(max_workers=self.max_parallel_clients) as executor:
                futures = [executor.submit(c.train) for c in self.selected_clients]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"[Warning] Client thread failed: {e}")

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()
            
            # ✅ 최적화된 함수 호출
            self.aggregate_and_send_heads()

            self.Budget.append(time.time() - s_t)
            print('-'*50, f"Round Time: {self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")

    def send_protos(self):
        assert (len(self.clients) > 0)
        # broadcast protos (simple logic)
        for client in self.clients:
            client.set_protos(self.global_protos)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_heads = []
        tot_samples = 0
        
        for client in active_clients:
            # 시간 비용 계산 생략 혹은 단순화 가능
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
            self.uploaded_heads.append(client.model.head)
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_and_send_heads(self):
        # 최적화된 solve_quadratic 사용
        head_weights = solve_quadratic(len(self.uploaded_ids), self.Vars, self.Hs)

        for idx, cid in enumerate(self.uploaded_ids):
            if head_weights[idx] is not None:
                new_head = self.add_heads(head_weights[idx])
            else:
                new_head = self.uploaded_heads[idx] # 인덱스 주의 (cid -> idx)

            self.clients[cid].set_head(new_head)

    def add_heads(self, weights):
        # 딥카피 오버헤드를 줄이기 위해 첫 번째 head 구조 활용
        new_head = copy.deepcopy(self.uploaded_heads[0])
        state_dict = new_head.state_dict()
        
        # State Dict 레벨에서 연산 (조금 더 빠름)
        for key in state_dict:
            state_dict[key].zero_()
        
        for w, head in zip(weights, self.uploaded_heads):
            if w <= 1e-8: continue # 작은 가중치 무시 (속도 향상)
            for key, param in head.state_dict().items():
                state_dict[key] += param * w
        
        new_head.load_state_dict(state_dict)
        return new_head


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            # stack -> mean이 더 빠름
            proto_stack = torch.stack(proto_list)
            agg_protos_label[label] = torch.mean(proto_stack, dim=0)
        else:
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label


def solve_quadratic(num_users, Vars, Hs):
    if num_users == 0:
        return []

    device = Hs[0].device
    tensor_Hs = torch.stack(Hs).reshape(num_users, -1).detach()
    
    # [방어 코드 1] Vars가 혹시 음수나 NaN인지 확인 및 보정
    Vars = [max(v, 1e-6) for v in Vars] # 최소값 클리핑
    tensor_Vars = torch.tensor(Vars, device=device, dtype=torch.float64)
    diag_Vars = torch.diag(tensor_Vars)

    avg_weight = []
    
    for i in range(num_users):
        diffs = tensor_Hs[i] - tensor_Hs 
        P_dist = torch.mm(diffs, diffs.T).to(torch.float64)
        P_gpu = diag_Vars + P_dist
        
        P = P_gpu.cpu().numpy()
        
        # Numerical Stability
        P = 0.5 * (P + P.T)
        P = np.nan_to_num(P, nan=0.0) # NaN 제거
        
        jitter = 1e-6 * (np.trace(P) / max(P.shape[0], 1) + 1.0)
        P = P + jitter * np.eye(P.shape[0], dtype=np.float64)

        # Solver 실행
        alpha = None
        try:
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, cvx.psd_wrap(P)))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            
            # OSQP가 실패하면 SCS 시도 (SCS가 좀 더 느리지만 튼튼함)
            try:
                prob.solve(solver=cvx.OSQP, verbose=False)
            except:
                prob.solve(solver=cvx.SCS, verbose=False)

            if alphav.value is not None:
                alpha = alphav.value
                # [방어 코드 2] 결과에 NaN이 섞여 있는지 확인
                if np.isnan(alpha).any():
                    alpha = None
        except Exception as e:
            # print(f"Solver failed: {e}")
            alpha = None

        # Solver가 실패했거나 NaN이 떴으면 -> "자기 자신"만 믿음 (Identity)
        if alpha is None:
            alpha = np.zeros(num_users)
            alpha[i] = 1.0
        else:
            # 작은 값 0 처리 + 정규화(합이 1이 되도록 안전장치)
            alpha[alpha < 1e-4] = 0.0
            alpha = alpha / (np.sum(alpha) + 1e-10) 

        avg_weight.append(alpha.tolist())

    return avg_weight
