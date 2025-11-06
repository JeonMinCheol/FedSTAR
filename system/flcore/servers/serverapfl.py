from flcore.clients.clientapfl import clientAPFL
from flcore.servers.serverbase import Server
from concurrent.futures import ThreadPoolExecutor, as_completed

class APFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAPFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.max_parallel_clients = 8


    def train(self):
        for i in range(1, self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
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

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAPFL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(i)
