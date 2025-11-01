
import threading,os,time

from datetime import datetime
from fedtorchPRO.superconfig import SuperConfig
try:
    import mlflow
except:
    from fedRecord.socket_imitate import mlflow

class Record:

    @staticmethod
    def create_runid(arguments,readbaselines = True):

        time.sleep(1)

        exname = SuperConfig.search_argment(arguments,'exname')
        if '/' in exname:
            exname = exname.split('/')[0]
        # Check if the experiment exists by name, otherwise create it
        experiment = mlflow.get_experiment_by_name(exname)

        if experiment is None or experiment.experiment_id == 'None':
            print(f"Experiment '{exname}' does not exist. Creating it.")
            experiment = mlflow.create_experiment(exname)
            experiment_id = experiment.experiment_id 
        else:
            experiment_id = experiment.experiment_id

        run_name = f"A{SuperConfig.search_argment(arguments,'alpha')}C{SuperConfig.search_argment(arguments,'clients_num')}({datetime.now().strftime("%Y_%m_%d_%H_%M_%S")})" 
        run = mlflow.start_run(run_name = run_name, experiment_id=experiment_id)
        if readbaselines:
            bsl = threading.Thread(target=Record.log_baselines,args=(arguments,))
            bsl.start()
        return  run.info.run_id,run_name
        
    def __init__(self, run_id):
        self.run = mlflow.start_run(run_id=run_id,nested = True)

    def log_param(self, key, value):
        # Log parameters to the current run
        mlflow.log_param(key, value)

    def end_run(self):
        # End the current run
        mlflow.end_run()

    def log_loss(self,name,cur_comic,loss):
        mlflow.log_metrics(step=cur_comic,metrics= {'los.' + name:loss,})

    def log_metric(self,name,cur_comic,metric,metric_type = 'acc'):
        mlflow.log_metrics(step=cur_comic,metrics= {"%s.%s"%(metric_type,name):metric,})
  
    @staticmethod
    def log_baselines(CONFIG):
        alpha, cn = str(CONFIG['workconfig']['alpha']).replace('.','d'),CONFIG['workconfig']['clients_num']
        
        path = os.path.join(
            SuperConfig.search_argment(CONFIG,'root_path'),
            SuperConfig.search_argment(CONFIG,'exname'),
            f'A{alpha}C{cn}','baselines'
        )

        os.makedirs(path,exist_ok=True)
        
        for methodname in os.listdir(os.path.join(path)):
            for metric_type in ['acc','los']:
                if not os.path.exists(os.path.join(path,methodname,f'{metric_type}.txt')):
                    continue
                with open(os.path.join(path,methodname,f'{metric_type}.txt'),'r',encoding='utf8') as fp:
                    data = eval(fp.read())
                    for i,item in enumerate(data):
                        c,v = item if type(item) == tuple else (i,item)                       
                        mlflow.log_metrics(step=c,metrics= {"%s.%s-B"%(metric_type,methodname):v,})

        mlflow.log_params(CONFIG)

    def print_exres(self,arguments):
        try:
            from fedtorchPRO.exp.result import extract_metrics,convert_to_pandas
            exres = extract_metrics(arguments.run_path)
            if arguments.read_baselines:
                exres.extend(extract_metrics(arguments.baselines_path))
            pd = convert_to_pandas(exres)
            if type(pd) == type(exres):
                for item in exres:
                    print(item)
            else:
                print(pd)
        except:
            pass