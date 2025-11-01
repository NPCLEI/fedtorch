
import torch

from torch_optimizer import Yogi,Lamb
from fedtorchPRO.core.utils  import Timer
from fedtorchPRO.core.server import Server
from fedtorchPRO.superconfig import Parser

class FedOPT(Server):

    def _init(self):

        self.opt = self.select_opt(self.global_model,self.arguments.server_lr)

        # 快捷开启MPX
        if '.MPX' in self.arguments.command:
            self.arguments['fedopt']['mpx'] = True
        if '.NAO' in self.arguments.command:
            self.arguments['fedopt']['nao'] = True

        if self.arguments.nao:
            self.o = self.arguments.NET()
            for tar,sou in zip(self.o.parameters(),self.global_model.parameters()):
                tar.data[:] = sou
            self.olr = self.arguments.fednao * self.arguments.server_lr
            self.naoopt = self.select_opt(self.o,self.olr)

    def select_opt(self,model,lr,optname = None):
        optname = optname if optname else self.arguments.fedopt['method']
        
        print(self.log_title,"FedOPT Optimizer is ",optname,'.')
        if 'grad' in optname:
            return torch.optim.Adagrad(model.parameters(),lr = lr)
        elif 'adam' in optname:
            return torch.optim.Adam(model.parameters(),lr = lr,amsgrad=self.arguments.ams,betas=self.arguments.get("betas",(0.9,0.999)))
        elif 'yogi' in optname:
            return Yogi(model.parameters(),lr = lr)
        elif 'lamb' in optname:
            return Lamb(model.parameters(),lr = lr,weight_decay=self.arguments.weight_decay)
        else:
            return torch.optim.SGD(model.parameters(),lr = lr,momentum=self.arguments.momentum)

    def _convert_FedAVG(self):
        self.arguments.config['server_arguments']['server_lr'] = 1
        self.arguments.config['fedopt']['method'] = 'sgd'
        self.arguments.config['fedopt']['momentum'] = 0
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config

    @torch.no_grad()
    def aggregate(self):
        """
            fedavg
        """

        weight = self.weight

        self.opt.zero_grad()

        gradnorm,count = 0,0
        for params in zip(
                self.global_model.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = (params[0].detach() - params[1]) * weight[0]
            for idx in range(2,len(params)):
                grad.add_(params[0].detach() - params[idx],alpha = weight[idx-1])
            params[0].grad = grad
            gradnorm += grad.norm()
            count += 1
        
        self.opt.step()
        self.grad_norm_record.append((gradnorm/count).item())
        
        return self.global_model
    
    def run(self):
        self._onrun()
        for self.cur_comic in range(self.arguments.communic_times):
            with Timer() as timer:
                self.select()
                self.train()
                self.aggregate()
                self.after_step()
                
            self._endround(timer)
        self._endrun()


    @torch.no_grad()
    def after_step(self):

        if self.arguments.nao:
            self.nao()

        if self.arguments.mpx:
            self.mpx()

    @torch.no_grad()
    def mpx(self):

        if self.arguments.keepnet:
            return self.mpxss()

        # beta = self.arguments.fedmpx / self.arguments.server_lr
        # alphas = [weight / self.arguments.client_trainer_arguments['lr'] for weight in self.weights]
            
        beta = self.arguments.fedmpx
        alphas = [0.5 * weight for weight in self.weight]

        gamma = 1.0 / (sum(alphas) + beta)

        for pms in zip(
            self.global_model.parameters(True),
            *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            p = pms[0]

            p.mul_(gamma * beta)
            
            for pm,alpha,weight in zip(pms[1:],alphas,self.weight):
                p.add_(pm , alpha = alpha * gamma)

    @torch.no_grad()
    def mpxss(self):

        last_ava_cids = self.selected_history[-2] if len(self.selected_history) > 1 else []
        last_ava_cids = [cid for cid in last_ava_cids if self[cid].losse_record[-1] < self.loss_record[-1]]

        cids = list(set(self.selected_history[-1] + last_ava_cids))
        weights = self.weights(cids)

        beta = self.arguments.fedmpx
        alphas = [0.5 * weight for weight in weights]

        gamma = 1.0 / (sum(alphas) + beta)

        for pms in zip(
            self.global_model.parameters(True),
            *[self[cid].model_.parameters(True) for cid in cids]
            ):
            p = pms[0]

            p.mul_(gamma * beta)
            
            for pm,alpha,weight in zip(pms[1:],alphas,self.weight):
                p.add_(pm , alpha = alpha * gamma)

    @torch.no_grad()
    def nao(self):

        self.naoopt.zero_grad()

        for params in zip(
                self.o.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = (params[0].detach() - params[1]) * self.weight[0]
            for idx in range(2,len(params)):
                grad.data[:] += (params[0].detach() - params[idx]) * self.weight[idx-1]
            params[0].grad = grad

        self.naoopt.step()

        slr = self.arguments.server_lr
        tau = 1/(1/self.olr + 1/slr)

        for g,o in zip(self.global_model.parameters(True),self.o.parameters(True)):

            g.mul_(tau / slr)
            
            g.add_(o, alpha = tau /self.olr)