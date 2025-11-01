
from fedtorchPRO.superconfig import SuperConfig
import sys,os,importlib

def _checkworkspace(CONFIG,baselines):
    
    runfilepath = sys.argv[0]
    folder_path = os.path.dirname(runfilepath)
    folder_name = os.path.basename(folder_path).lower()
    # A0d01C100
    alpha,clientn = folder_name.split('c')
    alpha = eval(alpha.replace('a','').replace('d','.'))
    clientn = eval(clientn)
    CONFIG['workconfig']['alpha'] =  alpha
    CONFIG['workconfig']['clients_num'] =  clientn
    # 检查数据是否存在
    expth = os.path.join(
        CONFIG['workconfig']['root_path'],
        CONFIG['workconfig']['exname'],
    )
    name = f"clientsn.{clientn}.alpha.{alpha}"
    datapth = os.path.join(expth,f"datas.map.{name}.txt")
    # 不存在，生成数据
    if not os.path.exists(datapth):
        print(f"数据映射文件不存在{datapth},开始生成数据。")
        from fedtorchPRO.data.utils import split_noniid_data
        # get_source = importlib.import_module(f'experiments.{CONFIG['workconfig']['exname']}.libs.dataset').get_source
        get_source = importlib.import_module(f'{CONFIG['workconfig']['libs']}dataset').get_source
        split_noniid_data(get_source(),SuperConfig(CONFIG))
    
    from fedtorchPRO.record import Record
    return Record.create_runid(CONFIG,readbaselines= baselines)

def process(SERVER,CONFIG,COMMD,RUNID,RUNNAME):
    if type(SERVER) == tuple:
        SERVER,CLIENT = SERVER
    else:
        CLIENT = None
    SERVER(SuperConfig(CONFIG,COMMD,RUNID,RUNNAME),CLIENT = CLIENT).run()

def auto_load_config():
    runfilepath = sys.argv[0]
    paths = runfilepath.split('/')
    return SuperConfig(importlib.import_module(f'{paths[-4]}.{paths[-3]}.{paths[-2]}.config').CONFIG)

def run(commands,baselines = True,CONFIG = 'auto',alpha = 'auto',clientn = 'auto'):
    if CONFIG == 'auto':
        from copy import deepcopy
        runfilepath = sys.argv[0]
        paths = runfilepath.split('/')
        CONFIG = importlib.import_module(f'{paths[-4]}.{paths[-3]}.{paths[-2]}.config').CONFIG
        CONFIG = deepcopy(CONFIG)
        CONFIG['workconfig']['read_baselines'] = baselines
    
    if alpha == 'auto' and clientn == 'auto':
        runid,runame = _checkworkspace(CONFIG,baselines)
    else:
        assert alpha != 'auto' and clientn != 'auto' , "Please give a spically alpha , clientn both."
        runid,runame = checkworkspace(CONFIG,baselines,alpha,clientn)
        commands = [(cmd[0],f".A{alpha}C{clientn} {cmd[1]}") for cmd in commands]

    for method,cmd in commands:
        process(method,CONFIG,cmd,runid,runame)

_created_runid = None

def checkworkspace(CONFIG,baselines,alpha,clientn):
    global _created_runid
    # A0d01C100
    runfilepath = sys.argv[0]
    paths = runfilepath.split('/')
    CONFIG['workconfig']['alpha'] =  alpha
    CONFIG['workconfig']['clients_num'] =  clientn
    CONFIG['workconfig']['exname'] = f"{CONFIG['workconfig']['exname']}/{paths[-2]}"  
    # 检查数据是否存在
    expth = os.path.join(
        CONFIG['workconfig']['root_path'],
        CONFIG['workconfig']['exname'],
    )
    name = f"clientsn.{clientn}.alpha.{alpha}"
    datapth = os.path.join(expth,f"datas.map.{name}.txt")
    # 不存在，生成数据
    if not os.path.exists(datapth):
        print(f"数据映射文件不存在{datapth},开始生成数据。")
        from fedtorchPRO.data.utils import split_noniid_data
        # get_source = importlib.import_module(f'experiments.{CONFIG['workconfig']['exname']}.libs.dataset').get_source
        get_source = importlib.import_module(f'{CONFIG['workconfig']['libs']}dataset').get_source
        split_noniid_data(get_source(),SuperConfig(CONFIG))
    
    if not _created_runid:
        from fedtorchPRO.record import Record
        _created_runid = Record.create_runid(CONFIG,readbaselines= baselines)
    else:
        from datetime import datetime
        # run.info.run_id,run_name
        run_id,run_name = _created_runid
        run_name = f"A{SuperConfig.search_argment(CONFIG,'alpha')}C{SuperConfig.search_argment(CONFIG,'clients_num')}({datetime.now().strftime("%Y_%m_%d_%H_%M_%S")})" 

    return run_id,run_name

