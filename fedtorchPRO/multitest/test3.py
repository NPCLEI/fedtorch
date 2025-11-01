
import sys
sys.path.append('./')



if __name__ == '__main__':
    from fedtorchPRO.core.client import training_function_debug
    from fedtorchPRO.core.cache import ServerUpdate,ProcessUpdate
    from fedtorchPRO.superconfig import SuperConfig
    from fedtorchPRO.core.server import Server
    import multiprocessing as mp
    
    from example2.config import CONFIG
    from example2.dataset import split_noniid_data
    
    scfg = SuperConfig(CONFIG)
    servercache = ServerUpdate(scfg.pknum)
    globalmodel = scfg.NET()
    servercache.statu_code = 1
    servercache.cur_comic = 0
    glm = ModelComicLink(scfg.NET())
    glm.update_shared_tensors(globalmodel)

    p = mp.Process(target=training_function_debug, args=(
        scfg,
        ProcessUpdate(scfg.clients_num),
        0,
        servercache,
        ModelGroupComicLink(scfg.NET,scfg.pknum),
        glm
    ))
    p.start()