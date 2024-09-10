from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging

from source.training.LRR import LRRLoss
from sklearn import metrics

class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler, lds, ld1, ld2):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:

            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            '''
            🧑‍💻
            The batch_size: 16
            The Task: ASD diagnosis on ABIDE
            😊
            # time_series   --- torch.Size([16, 200, 100]) 
            # node_feature  --- torch.Size([16, 200, 200])
            # predict       --- torch.Size([16, 2])
            # label         --- torch.Size([16, 2])
            '''

            out2lrr = node_feature.view(node_feature.size(0), -1)
            in_target = np.argmax(label.cpu().numpy(), axis=1)
            loss_lrr = LRRLoss()
            LRR_loss = loss_lrr.apply(out2lrr, in_target, lds)
            LRR_loss = torch.squeeze(LRR_loss)

            predict = self.model(time_series, node_feature)
            loss = self.loss_fn(predict, label)

            loss +=LRR_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter, lds, ld1, ld2):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            output = self.model(time_series, node_feature)
            label = label.float()

            out2lrr = node_feature.view(node_feature.size(0), -1)
            in_target = np.argmax(label.cpu().numpy(), axis=1)
            loss_lrr = LRRLoss()
            LRR_loss = loss_lrr.apply(out2lrr, in_target, lds)
            LRR_loss = torch.squeeze(LRR_loss)

            loss = self.loss_fn(output, label)

            loss +=LRR_loss

            loss_meter.update_with_weight(loss.item(), label.shape[0])

            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()
        
        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        
        f1_macro = metrics.f1_score(labels, result, average='macro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return [auc] + list(metric) + recall, f1_macro
    
    def generate_save_learnable_matrix(self):

        learable_matrixs = []
        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self, lds, ld1, ld2):

        training_process = []
        self.current_step = 0

        best_val_AUC = 0
        best_test_acc = 0
        best_test_AUC = 0
        best_test_sen = 0
        best_test_spec = 0
        best_test_recall = 0
        best_test_precision = 0
        best_test_f1_macro = 0

        for epoch in range(self.epochs):

            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0], lds, ld1, ld2)

            val_result, f1_macro = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy, lds, ld1, ld2)

            test_result, f1_macro = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy, lds, ld1, ld2)

            if epoch % 50 == 0:
                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'Test AUC:{test_result[0]:.4f}',
                    f'Test Sen:{test_result[-1]:.4f}'
                ]))

            if(val_result[0] > best_val_AUC):
                best_val_AUC = val_result[0]
                best_test_acc =  self.test_accuracy.avg
                best_test_AUC =  test_result[0]
                best_test_sen = test_result[-1]
                best_test_spec = test_result[-2]
                best_test_recall = test_result[-5]
                best_test_precision = test_result[-6]
                best_test_f1_macro = f1_macro

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'Test micro F1': test_result[-4],
                'Test micro recall': test_result[-5],
                'Test micro precision': test_result[-6],
                'Test f1_macro': f1_macro,
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        
        self.save_result(training_process)

        return [best_test_acc,best_test_AUC,best_test_sen,best_test_spec,\
                best_test_recall, best_test_precision,best_test_f1_macro]