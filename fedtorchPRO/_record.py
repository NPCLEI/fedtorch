
import mlflow,os,threading

from datetime import datetime
from fedtorchPRO.superconfig import SuperConfig

class Record:

    @staticmethod
    def create_runid(arguments,readbaselines = True):

        exname = SuperConfig.search_argment(arguments,'exname')
        # Check if the experiment exists by name, otherwise create it
        experiment = mlflow.get_experiment_by_name(exname)

        if experiment is None:
            print(f"Experiment '{exname}' does not exist. Creating it.")
            experiment_id = mlflow.create_experiment(exname)
        else:
            experiment_id = experiment.experiment_id

        run = mlflow.start_run(run_name = "%s"%datetime.now().strftime("%Y_%m_%d_%H_%M_%S") , experiment_id=experiment_id)
        if readbaselines:
            bsl = threading.Thread(target=Record.log_baselines,args=(arguments,))
            bsl.start()
        return  run.info.run_id
        
    def __init__(self, run_id):
        self.run = mlflow.start_run(run_id=run_id,nested = True)

    def log_param(self, key, value):
        # Log parameters to the current run
        mlflow.log_param(key, value)

    def end_run(self):
        # End the current run
        mlflow.end_run()

    def log_loss(self,name,cur_comic,loss):
        mlflow.log_metrics(
            step=cur_comic,
            metrics= {
                'loss.' + name:loss,
            }
        ) 

    def log_metric(self,name,cur_comic,metric,metric_type = 'acc'):
        mlflow.log_metrics(
            step=cur_comic,
            metrics= {
                "%s.%s"%(metric_type,name):metric,
            }
        )

    

    @staticmethod
    def log_baselines(CONFIG):
        path = os.path.join(
            SuperConfig.search_argment(CONFIG,'root_path'),
            SuperConfig.search_argment(CONFIG,'exname'),
            'baselines'
        )        

        if not os.path.exists(path):
            return
        for methodname in os.listdir(os.path.join(path)):
            for metric_type in ['acc','los']:
                if not os.path.exists(os.path.join(path,methodname,f'{metric_type}.txt')):
                    continue
                with open(os.path.join(path,methodname,f'{metric_type}.txt'),'r',encoding='utf8') as fp:
                    data = eval(fp.read())
                    for i,item in enumerate(data):
                        c,v = item if type(item) == tuple else (i,item)                       
                        mlflow.log_metrics(step=c,metrics= {"%s.%s"%(metric_type,methodname):v,})
