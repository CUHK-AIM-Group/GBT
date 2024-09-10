from datetime import datetime
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
import numpy as np

import time
from datetime import datetime

def model_training(cfg: DictConfig, lds, ld1, ld2):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)

    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    t_acc,t_auc,t_sen,t_spec, t_rec,t_pre,t_f1 = training.train(lds, ld1, ld2)
    return t_acc,t_auc,t_sen,t_spec, t_rec,t_pre, t_f1

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    count = 0
    acc_list = []
    auc_list = []
    sen_list = []
    spec_list = []
    rec_list = []
    pre_list = []
    f1_list = []

    for _ in range(cfg.repeat_time):

        ld1 = 1
        ld2 = 1
        lds = 0.000125
        t_acc,t_auc,t_sen,t_spec,t_rec,t_pre, t_f1 = model_training(cfg, lds, ld1, ld2)
        
        acc_list.append(t_acc)
        auc_list.append(t_auc)
        sen_list.append(t_sen)
        spec_list.append(t_spec)
        rec_list.append(t_rec)
        pre_list.append(t_pre)
        f1_list.append(t_f1)

        count = count + 1
    
    print("test acc mean {} std {}".format(np.mean(acc_list),np.std(acc_list)))
    print("test auc mean {} std {}".format(np.mean(auc_list)*100,np.std(auc_list)*100))
    print("test sensitivity mean {} std {}".format(np.mean(sen_list)*100,np.std(sen_list)*100))
    print("test specficity mean {} std {}".format(np.mean(spec_list)*100,np.std(spec_list)*100))
    print("test recall mean {} std {}".format(np.mean(rec_list)*100,np.std(rec_list)*100))
    print("test precision mean {} std {}".format(np.mean(pre_list)*100,np.std(pre_list)*100))
    print("test f1_macro mean {} std {}".format(np.mean(f1_list)*100,np.std(f1_list)*100))

if __name__ == '__main__':

    # Running time: start⏰
    tic = time.time()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    for i in range(1):
        main()

    # Running time: end⏰
    toc = time.time()
    print("⏰Time:", (toc - tic), "second \n", (toc - tic)/3600, "hour")

