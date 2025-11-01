import os
import torch
import random
import psutil
import threading

from datetime import datetime,timedelta

from fedtorchPRO.record import Record
from fedtorchPRO.core.client import Client
from fedtorchPRO.core.utils  import Timer
from fedtorchPRO.superconfig import SuperConfig
from fedtorchPRO.utils.npcprint import DualOutput

class Server:

    def __init__(self, arguments:SuperConfig,RECORD = None,CLIENT = None):
        self.print_capturer = DualOutput()
        self.mem = psutil.virtual_memory()
        self.test_records = []
        self.arguments = arguments
        self.arguments.server = self
        self.test_dataset = arguments.testdata
        """
            list : (client_id , length of client data set )
        """
        self.global_model = arguments.inimodel()
        self.available_gpus = arguments.available_gpus
        self.nk = {}
        self.time_recorder = []
        self.cur_comic = -1
        self.recorder = RECORD(arguments.run_id) if RECORD else Record(arguments.run_id)
        # 状态值存储
        self.loss_record  = []
        self.others = []
        self.selected_history = []
        self.grad_norm_record = []
        self.CLIENT = CLIENT if CLIENT else Client
        self._init()
        self._init_clients_()

    def _init(self):
        pass

    def _init_clients_(self):
        print(self.log_title,'CMD:',self.arguments.command)
        self.arguments.print_config()
        tt = threading.Thread(target=self.test)
        tt.start()
        datas = self.arguments.prepare_datas()
        print(self.log_title,'read data.')
        self.clients = [self.CLIENT(data,cid) for cid,data in enumerate(datas)]
        tt.join()

    def select(self):
        """
            参数：
            返回值:
        """

        if (self.cur_comic + 1) % self.arguments.test_interval  == 0:
            self.test()
            threading.Thread(target=self.save).start()
        if self.arguments.participation_rate_or_num == -1:
            self.selected_ids = [i for i in self.client_ids]
        else:
            random.shuffle(self.client_ids)
            self.selected_ids = self.client_ids[:self.arguments.pknum]
            print(self.log_title,self.selected_ids)
        self.selected_history.append(self.selected_ids)

    def train(self):
        # Debug
        if self.arguments.debug:
            for cid in self.selected_ids:
                self.clients[cid].TrainModel(self.arguments)
                print(self.log_title,cid,self[cid].losse_record[0],'->',self[cid].losse_record[-1],"Trained.")
            return
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.available_gpus)) as executor:
            futures = {executor.submit(self.clients[cid].TrainModel, self.arguments): cid for cid in self.selected_ids}
            for future in concurrent.futures.as_completed(futures):
                tid = futures[future]
                try:
                    future.result()  # 获取线程结果，确保异常被捕获
                    print(self.log_title,tid,self[tid].losse_record[0],'->',self[tid].losse_record[-1],"Trained.")
                except Exception as e:
                    print(f"Worker {tid} raised an exception: {e}")

    @torch.no_grad()
    def aggregate(self):
        """
            必须在本函数里一次性更改全局模型 , 返回模型的操作不能改变全局模型。
        """
        weights = self.weight

        for params in zip(self.global_model.parameters(True),*[self[cid].model.parameters(True) for cid in self.selected_ids]):
            params[0].data[:] = params[1].data * weights[0]
            for idx in range(2,len(params)):
                params[0].data.add_(params[idx].data, alpha = weights[idx-1])

        return self.global_model

    def test(self):
        if self.arguments.debug:
            return
        print(self.log_title , 'Testing...')
        acuv,metx = self.arguments.TestModel(
            net=self.global_model,
            test_dataset = self.test_dataset,
            commic_times = self.cur_comic,
            batch_size = self.arguments.test_batch_size,
            device = self.arguments.available_gpus[0].split('/')[0]
        )
        self.test_records.append((self.cur_comic,acuv))
        self.others.append((self.cur_comic,metx))        
        print(self.log_title , 'Tested.')
        if self.recorder:
            self.recorder.log_metric(self.arguments.methodname,self.cur_comic,acuv)

    def _onrun(self):
        self.available_client_nums = len(self.clients)
        self.client_ids = [i for i in range(self.available_client_nums)]

    def run(self):
        self._onrun()
        for self.cur_comic in range(self.arguments.communic_times):
            with Timer() as timer:
                self.select()
                self.train()
                self.aggregate()

            self._endround(timer)
        self._endrun()

    def _endrun(self):
        self.test()
        self.save()
        self.recorder.end_run()
        print()
        print(self.log_title,self.arguments.methodname,"END.")
        print("#"*30)
        self.recorder.print_exres(self.arguments)
        self.print_capturer.close()

    @property
    def method_path(self):
        pth = os.path.join(self.arguments.run_path,self.arguments.methodname)
        os.makedirs(pth,exist_ok=True)
        return pth

    def _save_lst(self,lst,name):
        with open(os.path.join(self.method_path,'%s.txt'%name),'w+',encoding='utf8') as fp:
            fp.write(str(lst))

    def save(self):
        os.makedirs(self.arguments.run_path,exist_ok=True)
        method_path = self.method_path
        os.makedirs(method_path,exist_ok=True)
        with open(os.path.join(method_path,'acc.txt'),'w+',encoding='utf8') as fp:
            fp.write(str(self.test_records))
        with open(os.path.join(method_path,'los.txt'),'w+',encoding='utf8') as fp:
            fp.write(str(self.loss_record))
        with open(os.path.join(method_path,'oth.txt'),'w+',encoding='utf8') as fp:
            fp.write(str(self.others))
        with open(os.path.join(method_path,'sel.txt'),'w+',encoding='utf8') as fp:
            fp.write(str(self.selected_history))
        with open(os.path.join(method_path,'cfg.txt'),'w+',encoding='utf8') as fp:
            self.arguments.config['command'] = self.arguments.command
            fp.write(str(self.arguments.config))
        if len(self.grad_norm_record):
            with open(os.path.join(method_path,'gdn.txt'),'w+',encoding='utf8') as fp:
                fp.write(str(self.grad_norm_record))
        self.print_capturer.save_output(os.path.join(method_path,'log.txt'))
        torch.save(self.global_model.state_dict(), os.path.join(method_path ,'model.states.pth'))
        print()
        print(self.log_title,"SAVED to",method_path)
        print()

    def _endround(self,timer):
        self.time_recorder.append(timer.elapsed_time)
        predict_interval = Timer.interval(self.time_recorder)
        tct = self.arguments.communic_times
        pdt_time = datetime.now() + timedelta(seconds = (tct - self.cur_comic - 1) * predict_interval)
        print(self.log_title,'This comicn time cost : %.5f sec.'%timer.elapsed_time,f'Predict end time :[ {pdt_time.strftime('%Y-%m-%d %H:%M:%S')} ( total : {tct *predict_interval/3600.0:2.2f} h) ]')
        avg_loss = 0
        for cid in self.selected_ids:
            # self.arguments.mdl_manager.release(self[cid].model,cid)
            self[cid].model = None
            avg_loss += self[cid].losse_record[0]
        self.loss_record.append(avg_loss/len(self.selected_ids))
        self.recorder.log_loss(self.arguments.methodname,self.cur_comic,self.loss_record[-1])

    @property
    def log_title(self):
        
        res = f'{self.__class__.__name__}/{self.CLIENT.__name__} : {self.cur_comic} rounds report '
        res = res + '[mem:%2.3f]'%(psutil.virtual_memory().used / (1024**3))
        return res
    
    @property
    def weight(self):
        if self.arguments.get('close_weight_by_samle_nums',False) or self.arguments.get('minibatch',False):
            return [1 for _ in self.selected_ids]
        nks = [self[cid].nk for cid in self.selected_ids]
        sum_ = sum(nks)
        return [i/sum_ for i in nks]
    
    def weights(self,cids):
        nks = [self[cid].nk for cid in cids]
        sum_ = sum(nks)
        return [i/sum_ for i in nks]
    
    def __getitem__(self,cid) -> Client:
        return self.clients[cid]

    def __len__(self):
        return len(self.clients)