import json
import logging
import argparse
import os
import sys
import random
import numpy
from transformers import AdamW
import  copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,ConcatDataset
from data_utils import   process_pt
import matplotlib.pyplot as plt
from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
from transformers import  AutoTokenizer

from model import PT_NT_DT, PT
import torch.nn.functional as F
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import pickle as pk

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert_name)
        self.tokenizer=tokenizer
        cache = 'cache/ssl_{0}_{1}.pk'.format(self.opt.dataset, self.opt.plm)
        if not os.path.exists(cache):
            labeled = process_pt(self.opt, self.opt.dataset_file['labeled'], tokenizer)
            test = process_pt(self.opt, self.opt.dataset_file['test'], tokenizer)
            unlabeled = process_pt(self.opt, self.opt.dataset_file['unlabeled'], tokenizer)
            dev = process_pt(self.opt, self.opt.dataset_file['dev'], tokenizer)
            if not os.path.exists('cache'):
                os.mkdir('cache')
            d = {'labeled': labeled, 'test': test, 'unlabeled': unlabeled, 'dev': dev}
            pk.dump(d, open(cache, 'wb'))

        d = pk.load(open(cache, 'rb'))
        self.labeled = d['labeled']
        self.unlabeled = d['unlabeled']
        self.testset = d['test']
        self.valset = d['dev']
        logger.info('labeled {}, unlabeled {}, test: {}, dev {}'.format(len(self.labeled),len(self.unlabeled),len(self.testset),len(self.valset)))




    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _evaluate_pt(self, model, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_inputs[-1]
                _, t_outputs, _ = model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true = t_targets_all.cpu().detach().numpy().tolist()
            pred = torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()

            f = metrics.f1_score(true, pred, average='macro', zero_division=0)
            r = metrics.recall_score(true, pred, average='macro', zero_division=0)
            p = metrics.precision_score(true, pred, average='macro', zero_division=0)
            acc = n_correct / n_total
            # acc= metrics.accuracy_score(true,pred)
            # error_rate = 1- metrics.zero_one_score(true,pred)
            # error_rate= metrics.accuracy_score(true, pred, normalize=False) / float(len(true))

            # tp, fn, fp, tn = metrics.confusion_matrix(true, pred).reshape(-1)
            error_rate = 0
            # error_rate= (fp + fn) / (tp + fn + fp + tn)
            # accu= (tp + tn) / (tp + fn + fp + tn)
            # logger.info(accu)
        return p, r, f, acc, error_rate

    def _evaluate_nt(self, model, criterion, val_data_loader):
        with torch.no_grad():
            pred_list, true_all = [], []
            test_loss = test_acc = 0.0
            # logger.info('testing')
            for i, v_sample_batched in enumerate(tqdm(val_data_loader)):
                labels = v_sample_batched['label']

                labels = labels.to(self.opt.device)

                inputs = [v_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _, logits, _ = model(inputs)

                loss = criterion(logits, labels)
                test_loss += inputs[0].size(0) * loss.data

                _, pred = torch.max(logits.data, -1)
                acc = float((pred == labels.data).sum())
                test_acc += acc
                pred_list.extend(pred.detach().cpu().tolist())
                true_all.extend(labels.data.detach().cpu().tolist())

            test_loss /= len(val_data_loader.dataset)
            test_acc /= len(val_data_loader.dataset)
            f1_sc = metrics.f1_score(true_all, pred_list, average='macro')
            f1_micro = metrics.f1_score(true_all, pred_list, average='micro')

            return test_loss, f1_sc, f1_micro, test_acc

    def _warm_up_plt(self,model, criterion,criterion_nr, optimizer, train_data_loader):

        global_step = 0

        epochs = self.opt.num_epoch
        with torch.no_grad():
            train_preds_hist = torch.zeros(len(train_data_loader.dataset), epochs)
            train_preds = torch.zeros(len(train_data_loader.dataset)) - 1.
            loss_total_p = []
            loss_total_n = []
            loss_total_each_c = []
            loss_total_each_n = []
        # train_losses = torch.zeros(len(train_data_loader.dataset)) - 1.
        for epoch in range(epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            loss_total=[]
            loss_each_c = []
            loss_each_n = []
            loss_p,loss_n=[],[]
            filter_acc=0
            truth=[]
            pred=[]
            model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                index = sample_batched['new_index']
                evid = sample_batched['is_evidence']
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,outputs,_= model(inputs)
                targets= inputs[-1]
                loss = criterion(outputs, targets)
                with torch.no_grad():
                    pred.extend(((criterion_nr(outputs, targets) > loss - torch.std(criterion_nr(outputs, targets))) == (
            evid.to(self.opt.device) == 0)).tolist())
                    truth.extend((evid == 0).tolist())
                    # filter_acc += ((criterion_nr(outputs, targets)>loss-torch.std(criterion_nr(outputs, targets))) ==(evid.to(self.opt.device)==0)).sum().data.item()

                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

                    if targets[evid == 1].size(0):
                        loss_each = criterion_nr(outputs[evid == 1], targets[evid == 1])
                        loss_each_c.extend(loss_each.cpu().detach().tolist())
                        loss_p.append(criterion(outputs[evid == 1], targets[evid == 1]).sum().detach().item())
                    if targets[evid == 0].size(0):
                        loss_each = criterion_nr(outputs[evid == 0], targets[evid == 0])
                        loss_each_n.extend(loss_each.cpu().detach().tolist())
                        loss_n.append(criterion(outputs[evid == 0], targets[evid == 0]).sum().detach().item())

                train_preds[index.cpu()] =criterion_nr(outputs,targets).cpu().data
            logger.info('filtering f1 {} acc {}'.format(metrics.f1_score(truth,pred, average='macro'), metrics.accuracy_score(truth, pred)))
            loss_total_p.append(np.mean(loss_p))
            loss_total_n.append(np.mean(loss_n))
            loss_total_each_c.append(loss_each_c)
            loss_total_each_n.append(loss_each_n)
            logger.info('epoch : {}'.format(epoch))

            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            # pres, recall, f1_score, acc, error_rate = self._evaluate_pt(model,val_data_loader)
            # logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f},  val_err: {:.4f}'.format(pres, recall, f1_score, acc, error_rate))

            train_preds_hist[:, epoch] = train_preds

        state = ({
            'train_preds_hist': train_preds_hist,
            'data': train_data_loader,
            'loss_total_p': loss_total_p,
            'loss_total_n': loss_total_n,
            'loss_total_each_c': loss_total_each_c,
            'loss_total_each_n': loss_total_each_n
        })
        fn_best='state_dict/{}-{}_warmup.tar'.format(self.opt.dataset, self.opt.noise_percentage)
        torch.save(state, fn_best)
        exit()
    def _warm_up__(self,model, criterion,criterion_nr, optimizer, train_data_loader,val_data_loader):
        labels_dict = json.load(open('../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))

        global_step = 0

        # epochs = self.opt.num_epoch//2
        epochs = self.opt.num_epoch
        # epochs = 12
        with torch.no_grad():
            train_preds_hist = torch.zeros(len(train_data_loader.dataset), epochs)
            train_preds = torch.zeros(len(train_data_loader.dataset)) - 1.

        # train_losses = torch.zeros(len(train_data_loader.dataset)) - 1.
        for epoch in range(epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            loss_total=[]
            loss_each_c = []
            loss_each_n = []
            loss_p,loss_n=[],[]
            model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                index = sample_batched['new_index']
                evid = sample_batched['is_evidence']
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,outputs,_= model(inputs)
                targets= inputs[-1]
                loss = criterion(outputs, targets)
                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

                    if targets[evid == 1].size(0):
                        loss_each = criterion_nr(outputs[evid == 1], targets[evid == 1])
                        loss_each_c.extend(loss_each.cpu().detach().tolist())
                        loss_p.append(criterion(outputs[evid == 1], targets[evid == 1]).sum().detach().item())
                    if targets[evid == 0].size(0):
                        loss_each = criterion_nr(outputs[evid == 0], targets[evid == 0])
                        loss_each_n.extend(loss_each.cpu().detach().tolist())
                        loss_n.append(criterion(outputs[evid == 0], targets[evid == 0]).sum().detach().item())

                train_preds[index.cpu()] =criterion_nr(outputs,targets).cpu().data

            logger.info('epoch : {}'.format(epoch))

            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            pres, recall, f1_score, acc, error_rate = self._evaluate_pt(model,val_data_loader)
            logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f},  val_err: {:.4f}'.format(pres, recall, f1_score, acc, error_rate))

            train_preds_hist[:, epoch] = train_preds

        # train_preds_hist_best =train_preds_hist [:,1]
        train_preds_hist_best =train_preds_hist.mean(1)
        mean, std = torch.mean(train_preds_hist_best), torch.std(train_preds_hist_best)
        data_filtered=[]
        for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
            texts = sample_batched['text']
            labels = sample_batched['label']
            labels_ori = sample_batched['ori_label']
            index = sample_batched['new_index']
            loss  = train_preds_hist_best[index]
            for i, indx in enumerate(index):
                evidenece=1 if loss[i] <mean else 0
                tem = dict()
                tem['text'] = texts[i]
                tem['ori_label'] = labels_dict[labels_ori[i].item()]
                tem['label'] = labels_dict[labels[i]]
                tem['is_evidence'] = evidenece
                data_filtered.append(tem)
        json.dump(data_filtered, open('data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.noise_percentage), 'w'), indent=3)
        filter_dataset_processed = process_pt(self.opt, 'data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.noise_percentage), self.tokenizer)
        for i in range(len(data_filtered)):
            filter_dataset_processed.data[i]['new_index'] = i
            filter_dataset_processed.data[i]['is_evidence'] = data_filtered[i]['is_evidence']
        os.remove('data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.noise_percentage))
        return filter_dataset_processed
    def _warm_up_old(self,model, criterion,criterion_nr, optimizer, train_data_loader,val_data_loader, use_polt=False):
        labels_dict = json.load(open('../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))

        global_step = 0
        # early_break = self.opt.num_epoch
        early_break = 1

        # epochs = self.opt.num_epoch//2
        epochs = self.opt.num_epoch
        # epochs = 12
        with torch.no_grad():
            train_preds_hist = torch.zeros(len(train_data_loader.dataset), epochs)
            train_preds = torch.zeros(len(train_data_loader.dataset)) - 1.
            loss_c_total, loss_n_total = [], []
        # train_losses = torch.zeros(len(train_data_loader.dataset)) - 1.
        loss_filter= 0
        for epoch in range(epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            loss_total=[]
            loss_each_c = 0
            loss_each_n = 0
            loss_c,loss_n=[],[]
            data_filtered=[]
            pred=[]
            truth=[]
            model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                index = sample_batched['new_index']
                evid = sample_batched['is_evidence']
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,outputs,_= model(inputs)
                targets= inputs[-1]
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    loss_nr= criterion_nr(outputs, targets)
                    # pred.extend(((loss_nr > loss - torch.std(loss_nr)) == (evid.to(self.opt.device) == 0)).tolist())

                    if loss.item() > 1.2:
                        pred.extend((loss_nr > loss - torch.std(loss_nr)).tolist())
                    else:
                        pred.extend((loss_nr > loss).tolist())
                    loss_each_n+=outputs[evid==0].size(0)
                    loss_each_c+=outputs[evid==1].size(0)
                    truth.extend((evid == 0).tolist())
                    loss_c.append(criterion(outputs[evid==1], targets[evid==1]).cpu().data.item())
                    loss_n.append(criterion(outputs[evid==0], targets[evid==0]).cpu().data.item())
                    # if epoch ==epochs-1:
                    if epoch ==early_break:
                        texts = sample_batched['text']
                        texts2 = sample_batched['text']
                        if self.opt.task in ['STS']:
                            texts2 = sample_batched['text2']
                        labels = sample_batched['label']
                        labels_ori = sample_batched['ori_label']
                        index = sample_batched['new_index']
                        loss_each = (loss_nr > loss - torch.std(loss_nr)).tolist()
                        # loss_each = (loss_nr > loss - torch.std(loss_nr)).tolist()
                        for i, indx in enumerate(index):
                            evidenece = 0 if loss_each[i]  else 1
                            # print(loss_each[i] , evidenece)
                            tem = dict()
                            tem['text'] = texts[i]
                            tem['text2'] = texts2[i]
                            tem['ori_label'] = labels_dict[labels_ori[i].item()]
                            tem['label'] = labels_dict[labels[i]]
                            tem['is_evidence'] = evidenece
                            data_filtered.append(tem)

                loss.sum().backward()

                optimizer.step()
                    # here save the filter data
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

            logger.info('epoch : {}'.format(epoch))
            loss_c_total.append(np.mean(loss_c))
            loss_n_total.append(np.mean(loss_n))
            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            logger.info('ground truth: clean {} noise {}'.format(loss_each_c, loss_each_n))
            pred_n = len([c for c in pred if c])
            pred_c = len([c for c in pred if not c])
            logger.info('filtering clean {} noisy {} f1 {} acc {}'.format(pred_c,pred_n,metrics.f1_score(truth,pred, average='macro'),  metrics.accuracy_score(truth, pred)))
            if epoch == early_break: break

        if use_polt:
            d_x = [str(i) for i in range(1, len(loss_c_total) + 1, 1)]
            plt.plot(d_x, loss_c_total, label='clean ')
            plt.plot(d_x, loss_n_total, label='noise')
            plt.legend(loc="lower right")
            plt.xlabel(' Epoch ')
            plt.title(' {} '.format(self.opt.noise_percentage))
            plt.ylabel('Loss')
            plt.show()
        # exit()
        json.dump(data_filtered, open('data_filtered_{}{}.json'.format(self.opt.dataset, self.opt.noise_percentage), 'w'), indent=3)
        filter_dataset_processed = process_pt(self.opt, 'data_filtered_{}{}.json'.format(self.opt.dataset, self.opt.noise_percentage), self.tokenizer)
        evid=[]
        for i in range(len(data_filtered)):
            filter_dataset_processed.data[i]['new_index'] = i
            filter_dataset_processed.data[i]['is_evidence'] = data_filtered[i]['is_evidence']
            evid.append(data_filtered[i]['is_evidence'])
        logger.info('clean {}, noisy {}'.format(len([_ for _ in evid if _ ==1]), len([_ for _ in evid if _ ==0])))
        os.remove('data_filtered_{}{}.json'.format(self.opt.dataset, self.opt.noise_percentage))
        return filter_dataset_processed
    def _warm_up(self,model, criterion,criterion_nr, optimizer, train_data_loader,val_data_loader):
        labels_dict = json.load(open('../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))

        global_step = 0

        # epochs = self.opt.num_epoch//2
        epochs = self.opt.num_epoch
        # epochs = 12
        with torch.no_grad():
            train_preds_hist = torch.zeros(len(train_data_loader.dataset), epochs)
            train_preds = torch.zeros(len(train_data_loader.dataset)) - 1.

        # train_losses = torch.zeros(len(train_data_loader.dataset)) - 1.
        for epoch in range(epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            loss_total=[]
            # loss_each_c = []
            # loss_each_n = []
            # loss_p,loss_n=[],[]
            data_filtered=[]
            pred=[]
            truth=[]
            model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                index = sample_batched['new_index']
                evid = sample_batched['is_evidence']
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,outputs,_= model(inputs)
                targets= inputs[-1]
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    loss_nr= criterion_nr(outputs, targets)
                    # pred.extend(((loss_nr > loss - torch.std(loss_nr)) == (evid.to(self.opt.device) == 0)).tolist())
                    pred.extend((loss_nr > loss - torch.std(loss_nr)).tolist())
                    # pred.extend((loss_nr > loss ).tolist())
                    truth.extend((evid == 0).tolist())

                    # if epoch ==epochs-1:
                    if epoch ==1:
                        texts = sample_batched['text']
                        labels = sample_batched['label']
                        labels_ori = sample_batched['ori_label']
                        index = sample_batched['new_index']
                        loss_each = (loss_nr > loss - torch.std(loss_nr)).tolist()
                        for i, indx in enumerate(index):
                            evidenece = 0 if loss_each[i]  else 1
                            # print(loss_each[i] , evidenece)
                            tem = dict()
                            tem['text'] = texts[i]
                            tem['ori_label'] = labels_dict[labels_ori[i].item()]
                            tem['label'] = labels_dict[labels[i]]
                            tem['is_evidence'] = evidenece
                            data_filtered.append(tem)

                loss.sum().backward()

                optimizer.step()
                    # here save the filter data
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

            logger.info('epoch : {}'.format(epoch))

            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            logger.info('filtering f1 {} acc {}'.format(metrics.f1_score(truth,pred, average='macro'), metrics.accuracy_score(truth, pred)))
            if epoch == 1: break
        json.dump(data_filtered, open('data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.noise_percentage), 'w'), indent=3)
        filter_dataset_processed = process_pt(self.opt, 'data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.noise_percentage), self.tokenizer)
        evid=[]
        for i in range(len(data_filtered)):
            filter_dataset_processed.data[i]['new_index'] = i
            filter_dataset_processed.data[i]['is_evidence'] = data_filtered[i]['is_evidence']
            evid.append(data_filtered[i]['is_evidence'])
        logger.info('clean {}, noisy {}'.format(len([_ for _ in evid if _ ==1]), len([_ for _ in evid if _ ==0])))
        os.remove('data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.noise_percentage))
        return filter_dataset_processed


    def _train_pt(self,model, criterion, optimizer, train_data_loader, val_data_loader,test_data_loader, save=False, warming=False ):
        max_val_acc = 0
        global_step = 0
        earlier_out = 0
        path = None
        num_epoch = self.opt.num_epoch_warming if warming else self.opt.num_epoch
        # if (self.opt.train_sample in [30] and save) or self.opt.dataset not in ['AG', 'yelp', 'yahoo']:
        #     num_epoch= 60

        t_total= int(len(train_data_loader) * num_epoch)


        for epoch in range(num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            loss_total=[]
            model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,outputs,_= model(inputs)
                targets= inputs[-1]
                # print(outputs)
                # print(targets)
                # print('--------------------')
                loss = criterion(outputs, targets)
                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

            logger.info('epoch : {}'.format(epoch))

            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            pres, recall, f1_score, acc, error_rate = self._evaluate_pt(model,val_data_loader)
            # logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f},  val_err: {:.4f}'.format(pres, recall, f1_score, acc, error_rate))

            if f1_score > max_val_acc:
                earlier_out= 0
                max_val_acc = f1_score
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = copy.deepcopy(model.state_dict())
            else:
                earlier_out+=1


        model.load_state_dict(path)
        model.eval()
        if self.opt.dataset in ['TNEWS', 'OCNLI', 'IFLYTEK']:
            logger.info('save model for chinese dataser')
            path = 'state_dict/{}_{}_{}_scenario_{}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.noise_percentage),self.opt.scenario )
            torch.save(model.module.state_dict(), path)
        if save:
            return model.state_dict()
        else:
            with open('results/main_sstc.txt', 'a+') as f:
                f.write(' {} final pt {} f1  {} acc {}   \n'.format(self.opt.dataset, self.opt.train_sample,
                                                                    str(round(f1_score, 4)), str(round(acc, 4))))
            f.close()
    def _train_nt(self,model,optimizer,weight,criterion_noise,criterion,criterion_nll,criterion_nr, train_data_loader, val_data_loader, test_data_loader, save_model=True):


        t_total= int(len(train_data_loader) * self.opt.num_epoch_negative)


        best_acc_test=0
        global_step = 0
        best_test_acc = 0.0
        best_f1_micro_test = 0.0
        best_valid_acc = 0.0
        best_f1_test = 0.0
        earlier_out = 0
        train_losses = torch.zeros(len(train_data_loader.dataset)) - 1.
        len_dataloader= len(train_data_loader.dataset)
        with torch.no_grad():

            clean_ids = [d['new_index'] for _, d in enumerate(train_data_loader.dataset) if d['is_evidence'] == 1]
            noisy_ids = [d['new_index'] for _, d in enumerate(train_data_loader.dataset) if d['is_evidence'] == 0]

            train_preds_hist = torch.zeros(len(train_data_loader.dataset), self.opt.num_hist, self.opt.lebel_dim)
            train_preds = torch.zeros(len(train_data_loader.dataset), self.opt.lebel_dim) - 1.
            clean_labels = [d['label'] for d in train_data_loader.dataset if d['is_evidence'] == 1]
        for epoch in range(self.opt.num_epoch_negative):
            train_loss =train_rev= train_loss_neg = train_acc = 0.0
            pl = 0.
            nl = 0.
            model.train()
            if epoch % self.opt.num_hist == 0 and epoch != 0:
                # if epoch in self.opt.epoch_step:

                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total,
                                                                           self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                    self.opt.learning_rate = param_group['lr']

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):

                model.zero_grad()

                global_step += 1
                optimizer.zero_grad()
                labels = sample_batched['label']
                index = sample_batched['new_index']
                evid = sample_batched['is_evidence']
                evid=evid.to(self.opt.device)


                p = float(global_step + epoch * len_dataloader) / \
                    self.opt.num_epoch_negative  / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # index_noise, index_clean= index[evid == 0], index[evid == 1]

                # with torch.no_grad():
                label_clean, label_noise = labels[evid == 1], labels[evid == 0]
                labels_neg = (label_noise.unsqueeze(-1).repeat(1, self.opt.neg_sample_num) + torch.LongTensor(len(label_noise),self.opt.neg_sample_num).random_(1, self.opt.lebel_dim)) % self.opt.lebel_dim
                labels = labels.to(self.opt.device)
                label_clean = label_clean.to(self.opt.device)
                labels_neg = labels_neg.to(self.opt.device)

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,logits,logits_rev = model(inputs, alpha)

                loss_rev = criterion_noise(logits_rev, evid)

                logits_clean , logits_noise = logits[evid==1], logits[evid==0]
                ##
                # with torch.no_grad():
                s_neg = torch.log(torch.clamp(1. - F.softmax(logits_noise, -1), min=1e-5, max=1.))
                s_neg *= weight[label_noise].unsqueeze(-1).expand(s_neg.size()).cuda()

                _, pred = torch.max(logits.data, -1)
                acc = float((pred == labels.data).sum())
                train_acc += acc

                # train_loss += inputs[0].size(0) * criterion(logits, labels).data
                # train_loss_neg += inputs[0].size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data
                # train_losses[index] = criterion_nr(logits, labels).cpu().data

                if logits_clean.size(0):
                    train_loss += logits_clean.size(0) * criterion(logits_clean, label_clean).data
                train_rev += evid.size(0) * loss_rev.data
                # train_loss_neg += inputs[0].size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data
                train_loss_neg += logits_noise.size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data
                train_losses[index] = criterion_nr(logits, labels).cpu().data

                ##

                if epoch >= self.opt.switch_epoch:
                    if epoch == self.opt.switch_epoch and i_batch == 0: logger.info('Switch to GNT')
                    labels_neg[train_preds_hist.mean(1)[index[evid==0], label_noise] < 1 / float(self.opt.lebel_dim)] = -100
                    # labels_neg[train_preds_hist.mean(1)[index, labels] < 1 / float(self.opt.lebel_dim)] = -100
                    # labels_neg[train_preds_hist.mean(1)[index, labels] < .8] = -100

                labels = labels * 0 -100
                loss = criterion(logits_clean, label_clean) * float((label_clean >= 0).sum())
                # loss = criterion(logits, labels) * float((labels >= 0).sum())
                loss_neg = criterion_nll(s_neg.repeat(self.opt.neg_sample_num, 1),
                                              labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
                # ((loss + loss_neg) / (float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum()))).backward()
                # ((loss + loss_neg) / (float((label_clean >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum()))).backward()
                if self.opt.use_ads:
                    ((loss_rev+loss + loss_neg) / (float((evid >= 0).sum())+ float((label_clean >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum()))).backward()
                else:
                    ((loss + loss_neg) / (float((label_clean >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum()))).backward()

                optimizer.step()
                #
                # with torch.no_grad():
                # train_preds[index[evid == 0]] = F.softmax(logits_noise, -1).cpu().data
                train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data
                pl += float((labels >= 0).sum())
                nl += float((labels_neg[:, 0] >= 0).sum())

            # with torch.no_grad():
            train_loss /= len(clean_ids)
            # train_loss /= len(train_data_loader.dataset)
            train_rev /= len(train_data_loader.dataset)
            # train_loss_neg /= len(train_data_loader.dataset)
            train_loss_neg /= len(noisy_ids)
            train_acc /= len(train_data_loader.dataset)
            pl_ratio = pl / float(len(train_data_loader.dataset))
            nl_ratio = nl / float(len(train_data_loader.dataset))
            noise_ratio = 1. - pl_ratio

            # noise = (np.array(trainset.imgs)[:, 1].astype(int) != np.array(clean_labels)).sum()
            # (np.array([d['label'] for d in train_data_loader.dataset]) != np.array(clean_labels))
            noise = len(noisy_ids)
            logger.info(
                '[%6d/%6d] loss: %5f,train_rev: %5f, loss_neg: %5f, acc: %5f, lr: %7f, noise: %d, pl: %5f, nl: %5f, noise_ratio: %5f'
                % (epoch, self.opt.num_epoch_negative, train_loss, train_rev, train_loss_neg, train_acc, self.opt.learning_rate, noise,
                   pl_ratio, nl_ratio,
                   noise_ratio))


            model.eval()
            with torch.no_grad():
                logger.info('validating')
                val_loss, val_f1_sc,val_f1_micro, val_acc = self._evaluate_nt(model, criterion, val_data_loader)
                # best_test_acc = max(val_f1_sc, best_test_acc)
                # best_test_acc = max(val_acc, best_test_acc)
                best_valid_acc = max(val_acc, best_valid_acc)

                logger.info('\t valid ...loss: %5f, acc: %5f,f1: %5f,f1 micro: %5f, best_acc: %5f' % (
                    val_loss, val_acc, val_f1_sc,val_f1_micro,best_valid_acc))

            is_best = val_acc >= best_valid_acc
            if is_best:
                earlier_out = 0
                path = 'state_dict/{0}_nt_{1}_{2}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample))
                if self.opt.save_model_nt:
                    torch.save(model.module.state_dict(), path)
                else:

                    model.eval()
                    with torch.no_grad():
                        logger.info('testing')
                        # test_loss, f1_sc, f1_micro, test_acc
                        test_loss, test_f1_sc,test_f1_micro, test_acc = self._evaluate_nt(model, criterion,test_data_loader)
                        best_acc_test = max(best_acc_test, test_acc)
                        best_f1_test = max(best_f1_test, test_f1_sc)
                        best_f1_micro_test = max(best_f1_micro_test, test_f1_micro)
                        logger.info('\t test ...loss: %5f, acc: %5f,f1 macro: %5f , f1 micro: %5f best_acc: %5f best_f1: %5f best_f1 micro: %5f ' % (
                            test_loss, test_acc, test_f1_sc,test_f1_micro,best_acc_test ,best_f1_test, best_f1_micro_test ))

            ##
            else:
                earlier_out += 1
            inds = np.argsort(np.array(train_losses))[::-1]
            rnge = int(len(val_data_loader.dataset) * noise_ratio)
            # inds_filt = inds[:rnge]

            model.train()

            assert train_preds[train_preds < 0].nelement() == 0
            train_preds_hist[:, epoch % self.opt.num_hist] = train_preds
            train_preds = train_preds * 0 - 1.
            assert train_losses[train_losses < 0].nelement() == 0
            train_losses = train_losses * 0 - 1.


            if True and save_model and epoch % self.opt.num_hist == 0:

                logger.info('saving separated histogram...')
                plt.hist(train_preds_hist.mean(1)[torch.arange(len(train_data_loader.dataset))[clean_ids]
                                                  , np.array([d['label'] for d in train_data_loader.dataset])[
                    clean_ids]].numpy(), bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean')
                plt.hist(train_preds_hist.mean(1)[
                             torch.arange(len(train_data_loader.dataset))[noisy_ids]
                             ,  np.array([d['label'] for d in train_data_loader.dataset])[
                                 noisy_ids]].numpy(), bins=20, range=(0., 1.), edgecolor='black', alpha=
                         0.5, label='noisy')
                plt.xlabel('probability')
                plt.ylabel('number of data')
                plt.grid()
                plt.savefig(self.opt.save_dir_histo + '/histogram-SSTC-pt_label-{}-{}-{}.jpg'.format(self.opt.dataset, self.opt.train_sample, epoch))
                plt.clf()
            if epoch >= self.opt.switch_epoch and earlier_out >3: break

        state = ({
            'epoch' : epoch,
            # 'state_dict' 	  : model.state_dict(),
            'data': train_data_loader,
            # 'optimizer' 	  : optimizer.state_dict(),
            'train_preds_hist': train_preds_hist,
            'lr': self.opt.learning_rate,
            'pl_ratio' 		  : pl_ratio,
            'nl_ratio' 		  : nl_ratio,
        })
        fn_best='state_dict/_{}-{}_ssl_pt_label.tar'.format(self.opt.dataset, self.opt.train_sample)
        if save_model:
            torch.save(state, fn_best)
        with open('results/main_sstc.txt', 'a+') as f :
            f.write(' {} nt {} f1  {} acc {}   \n'.format(self.opt.dataset, self.opt.train_sample, str(round(best_f1_test, 4)), str(round(best_acc_test, 4))))
        f.close()

    def reinitialization(self, trainset,testset,valset, PT_train=False, warming=False):
        self.opt.learning_rate= 1e-5
        batch_size= self.opt.batch_size
        if warming and self.opt.train_sample ==30:
            batch_size=8
        # if len(trainset) <1000:
        #     batch_size =16

        # num_epoch =  self.opt.num_epoch if PT  else self.opt.num_epoch_negative
        # if self.opt.train_sample  in [30] and warmup:
        #     batch_size = 4
        #     num_epoch = 30

        seed = random.randint(20, 300)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




        train_data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.batch_size_val, shuffle=False)
        val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size_val, shuffle=False)
        # t_total = int(len(train_data_loader) * num_epoch)

        # self.refining(train_data_loader)
        # if warmup:
        #     model = Baseline_pt(self.opt)
        # else:
        #     model = Pure_Roberta(self.opt)
        if PT_train:
            model = PT(self.opt)
        else:
            model = PT_NT_DT(self.opt)
        model = nn.DataParallel(model)
        model.to(self.opt.device)
        _params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self.opt.optimizer(model.parameters(), lr=self.opt.learning_rate,
                                       weight_decay=self.opt.l2reg)

        weight = torch.FloatTensor(self.opt.lebel_dim).zero_() + 1.
        for i in range(self.opt.lebel_dim):
            weight[i] = (torch.from_numpy(
                np.array([d['label'] for d in trainset]).astype(int)) == i).sum()
        weight = 1 / (weight / weight.max())

        weight = weight.to(self.opt.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        criterion_nll = nn.NLLLoss()
        criterion_nr = nn.CrossEntropyLoss(reduce=False)
        criterion_noise = nn.CrossEntropyLoss()

        return train_data_loader, test_data_loader, val_data_loader, model, optimizer, weight, criterion, criterion_nll, criterion_nr, criterion_noise

    def _warmup_training(self, model, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader, unlabeledset):

        states = self._train_pt(model, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader,
                       save=True, warming=True)

        # t_total = int(len(train_data_loader) * num_epoch)

        model.load_state_dict(states)
        model.eval()
        logger.info('generate pseudo-labels')

        unlabeled_data_loader = DataLoader(dataset=unlabeledset, batch_size=self.opt.batch_size_val, shuffle=False)
        # path = '../state_dict/{0}_{1}_{2}_new.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample))

        # model = Baseline_pt(self.opt)
        # model.to(self.opt.device)


        n_correct=n_total = 0
        t_targets_all= t_outputs_all = None
        model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(unlabeled_data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['ori_label'].to(self.opt.device)
                _, t_outputs,_ = model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true = t_targets_all.cpu().detach().numpy().tolist()
            pred = torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()


            f1 = metrics.f1_score(true, pred, average='macro', zero_division=0)
            rec = metrics.recall_score(true, pred, average='macro', zero_division=0)
            pres = metrics.precision_score(true, pred, average='macro', zero_division=0)

        logger.info('pseudo-labels performance ')
        logger.info('rec {}, prec, {}, f1 {}'.format(round(rec, 4), round(pres, 4), round(f1, 4)))

        return pred
    def refining(self, train_data_loader=None, use_plt=False):
        labels_dict = json.load(open('../../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))
        fn_best = 'state_dict/_{}-{}_ssl_pt_label.tar'.format(self.opt.dataset, self.opt.train_sample)
        # fn_best = 'state_dict/_{}-{}_ssl.tar'.format(self.opt.dataset, self.opt.train_sample)
        # fn_best = 'state_dict/_{}-{}_ssl.tar'.format(self.opt.dataset, self.opt.train_sample)
        states = torch.load(fn_best)
        train_preds_hist = states['train_preds_hist']
        train_data_loader = states['data']
        nl_ratio = states['nl_ratio']
        logger.info('nl_ratio {}'.format(nl_ratio))
        clean_ids = [d['new_index'] for _, d in enumerate(train_data_loader.dataset) if d['is_evidence'] == 1]
        noisy_ids = [d['new_index'] for _, d in enumerate(train_data_loader.dataset) if d['is_evidence'] == 0]
        logger.info('clean {} noise {}'.format(len(clean_ids), len(noisy_ids)))
        # train_data_loader = states['data']
        acc=t_noisy=0.0
        data_filtered=[]
        if use_plt:
            all_index =all_evid=all_labels= torch.tensor([], dtype=torch.long)
            for sample_batched in tqdm(train_data_loader):
                all_index = torch.cat((all_index, sample_batched['new_index']), dim=0)
                all_evid = torch.cat((all_evid, sample_batched['is_evidence']), dim=0)
                all_labels = torch.cat((all_labels, sample_batched['label']), dim=0)

            plt.hist(train_preds_hist.mean(1)[all_index[all_evid==1], all_labels[all_evid==1] ].numpy(), bins=20, range=(0., 1.), edgecolor='black',
                     alpha=0.5, label='clean')
            plt.hist(train_preds_hist.mean(1)[all_index[all_evid==0], all_labels[all_evid==0] ].numpy(), bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy')
            plt.xlabel('probability');
            plt.ylabel('number of data')
            plt.grid()
            plt.show()
            plt.clf()
            print()

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                    texts = sample_batched['text']
                    labels = sample_batched['label']
                    labels_ori = sample_batched['ori_label']
                    index = sample_batched['new_index']
                    evid = sample_batched['is_evidence']
                    noise_labels = labels[evid == 0]
                    clean_labels = labels[evid == 1]
                    if not labels[evid == 0].size(0): continue
                    (labels[evid == 1] == torch.max(train_preds_hist.mean(1)[index[evid == 1]], dim=-1)[1]).sum()
                    filtered = train_preds_hist.mean(1)[index[evid == 0], noise_labels] < .5
                    if not filtered.sum(): continue
                    pro, pred = torch.max(train_preds_hist.mean(1)[index[evid == 0]][filtered], dim=-1)
                    filtered_achieve_thres = pro > .9
                    if not filtered_achieve_thres.sum(): continue
                    acc += (pred[filtered_achieve_thres] == labels_ori[evid == 0][filtered][filtered_achieve_thres]).sum()
                    t_noisy += labels_ori[evid == 0][filtered][filtered_achieve_thres].size(0)

                    index_to_save = torch.cat([index[evid == 1], index[evid == 0][filtered][filtered_achieve_thres]])
                    for i, indx in enumerate(index):
                        if indx not in index_to_save: continue
                        tem = dict()
                        tem['text'] = texts[i]
                        tem['ori_label'] = labels_ori[i]
                        tem['ori_label'] = labels_ori[i]
                    print(t_noisy)
                    print(acc / t_noisy)
        clean_averaged = []
        for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
            labels = sample_batched['label']
            index = sample_batched['new_index']
            evid = sample_batched['is_evidence']
            clean_averaged.extend(train_preds_hist.mean(1)[index[evid==1], labels[evid==1]].tolist())

        prob_average= np.mean(clean_averaged)
        logger.info('averaged clean prob {}'.format(prob_average))
        for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
            texts = sample_batched['text']
            labels = sample_batched['label']
            labels_ori = sample_batched['ori_label']
            index = sample_batched['new_index']
            evid = sample_batched['is_evidence']
            index_noise = index[evid==0]
            pro_l = train_preds_hist.mean(1)[index, labels]
            pro, pred = torch.max(train_preds_hist.mean(1)[index], dim=-1)
            for i, indx in enumerate(index):
                evidenece = evid[i].item()
                label = labels[i].item()
                if evidenece ==0:
                    # label = pred[i].item()

                    # if pro_l[i] <nl_ratio and  pro[i]>.9:
                    # if pro_l[i] <nl_ratio and  pro[i]>1-nl_ratio:
                    # if pro_l[i] <nl_ratio and  pro[i]>1-nl_ratio + (1/self.opt.lebel_dim):
                    # if pro_l[i] <nl_ratio and  pro[i]>prob_average:
                    # if pro_l[i] <(1/self.opt.lebel_dim):
                    # if pro_l[i] <.1  and  pro[i]>prob_average:
                    # if pro[i]>.95:
                    # if label!=pred[i].item()  :
                    if pro[i]>.95  :
                        label = pred[i].item()
                        # evidenece=1
                    else:
                        continue
                # else:
                #     continue

                tem = dict()
                tem['text'] = texts[i]
                tem['ori_label'] = labels_dict[labels_ori[i].item()]
                tem['label'] = labels_dict[label]
                tem['is_evidence'] = evidenece
                data_filtered.append(tem)

        try:
            # true_label = [d['ori_label'] for d in data_filtered if d['is_evidence']==1]
            true_label = [d['ori_label'] for d in data_filtered ]
            # pred_label = [d['label'] for d in data_filtered if d['is_evidence']==1]
            pred_label = [d['label'] for d in data_filtered ]
            logger.info('------'*5)
            logger.info('filtering performance ')
            logger.info('acc {}   f1 macro {} '.format(metrics.accuracy_score(true_label,pred_label ), metrics.f1_score(true_label,pred_label, average='macro' )))
            logger.info(metrics.classification_report(true_label, pred_label))
            logger.info('------' * 5)
        except:
            pass

        #save
        #
        json.dump(data_filtered, open('data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.train_sample), 'w'), indent=3)
        filter_dataset_processed = process_pt(self.opt,'data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.train_sample), self.tokenizer)
        for i in range(len(data_filtered)):
            filter_dataset_processed.data[i]['is_evidence'] = data_filtered[i]['is_evidence']
            filter_dataset_processed.data[i]['new_index'] = i
        os.remove('data_filtered-{}-{}.json'.format(self.opt.dataset, self.opt.train_sample))
        logging.info('training after cleaning  {}'.format(len(filter_dataset_processed.data)))
        return filter_dataset_processed

    def run(self):

        labeledset = self.labeled
        unlabeledset = self.unlabeled
        testset = self.testset
        valset = self.valset

        # train on N labeled instances
        if self.opt.train_sample > 0 and len(labeledset) > self.opt.train_sample:
            if self.opt.train_sample in [30]:
                labeled_labels = np.array([v['label'] for v in labeledset])
                train_labeled_idxs, _ = train_test_split(list(range(len(labeled_labels))),
                                                         train_size=self.opt.train_sample * self.opt.lebel_dim,
                                                         stratify=labeled_labels)
                labeledset.data = [labeledset[i] for i in train_labeled_idxs]
            else:
                index = random.sample(range(0, len(labeledset)), self.opt.train_sample)
                labeledset = torch.utils.data.Subset(labeledset, index)
        for i in range(len(labeledset)):
            labeledset[i]['is_evidence'] = 1

        logger.info('train labeled {} unlabeled {} val {} test {}'.format(len(labeledset), len(unlabeledset), len(valset), len(testset)))
        train_data_loader, test_data_loader, val_data_loader, model, optimizer, weight, criterion, _, _, _=self.reinitialization(labeledset, testset, valset, PT_train=True, warming=True)

        logger.info('Pseudo-labels assignment  ')
        pseudo_labels = self._warmup_training(model, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader, unlabeledset)
        for i in range(len(unlabeledset)):
            unlabeledset[i]['label'] = pseudo_labels[i]
            unlabeledset[i]['is_evidence'] = 0
        trainset = ConcatDataset([labeledset, unlabeledset])
        for i in range(len(trainset)):
            trainset[i]['new_index'] = i

        logger.info('train {} labeled {}, unlabled {}, test {}, dev {}'.format(len(trainset),len(labeledset), len(unlabeledset), len(testset), len(valset)))

        train_data_loader, test_data_loader, val_data_loader, model, optimizer, weight, criterion, criterion_nll, criterion_nr, criterion_noise=self.reinitialization(trainset, testset, valset)

        self._train_nt(model,optimizer,weight,criterion_noise,criterion,criterion_nll,criterion_nr,  train_data_loader, val_data_loader, test_data_loader)
        logger.info('refining ')
        trainset_filtered = self.refining()
        train_data_loader_filtered, test_data_loader, val_data_loader, model, optimizer, weight, criterion, criterion_nll, criterion_nr, criterion_noise=self.reinitialization(trainset_filtered, testset, valset, PT_train=True)
        logger.info('final fine-tuning using pt')
        self._train_pt( model, criterion, optimizer, train_data_loader_filtered, val_data_loader, test_data_loader)



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AG', required=True, type=str, help=' AG, yelp, yahoo ')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')
    parser.add_argument('--num_epoch_warming', default=50, type=int, help='')
    parser.add_argument('--num_epoch', default=10, type=int, help='')
    parser.add_argument('--num_epoch_negative', default=13, type=int, help='')
    parser.add_argument('--switch_epoch', default=7, type=int, help='')
    parser.add_argument('--num_hist', default=2, type=int, help='')
    parser.add_argument('--neg_sample_num', default=10, type=int, help='')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--batch_size_val', default=256, type=int, help='')
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.002, type=float)
    parser.add_argument('--use_noisy', default=0, type=int, help='0 false or 1 true')
    parser.add_argument('--plm', default='bert', type=str, help='0 false or 1 true')
    parser.add_argument('--train_sample', default=0, type=int, help='0 false or 1 true')
    parser.add_argument('--save_model_nt', default=0, type=int, help='0 false or 1 true')
    parser.add_argument('--use_ads', default=1, type=int, help='0 false or 1 true')
    parser.add_argument('--use_pseudo_label', default=1, type=int, help='0 false or 1 true')
    parser.add_argument('--noise_percentage', default=0.0, type=float, help='0 false or 1 true')
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--save_dir', default='state_dict' , type=str, help='e.g. cuda:0')
    parser.add_argument('--device_group', default='4' , type=str, help='e.g. cuda:0')
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--seed', default=65, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()

    if opt.dataset in ['TNEWS', 'IFLYTEK', 'OCNLI', 'AFQMC']:
        opt.plm='chinese'
        opt.save_model_nt=1


    label_dims = {'TNEWS': 15, 'OCNLI': 3, 'IFLYTEK': 119, 'yelp': 5, 'AFQMC': 2, 'IMDB': 2, 'semeval': 2,
                  'semeval16_rest': 2, 'sentihood': 4, 'TREC': 6, 'DBPedia': 14, 'AG': 4,
                  'SUBJ': 2, 'ELEC': 2, 'SST': 2, 'SST-5': 5, 'CR': 2, 'MR': 2, 'PC': 2, 'yahoo': 10, 'MPQA': 2,
                  'R8': 8, 'hsumed': 23}
    opt.lebel_dim = label_dims[opt.dataset]
    opt.max_seq_len = {'TNEWS': 128, 'OCNLI': 128, 'IFLYTEK': 128, 'AFQMC': 128, 'yelp': 256, 'TREC': 20, 'yahoo': 256,
                       'ELEC': 256, 'MPQA': 10, 'AG': 100, 'MR': 30, 'SST': 30, 'SST-5': 30, 'PC': 30, 'CR': 30,
                       'DBPedia': 160, 'IMDB': 280, 'SUBJ': 30,
                       'semeval': 80, 'R8': 207, 'hsumed': 156}.get(opt.dataset)
    task_list = {'CLUE': ['TNEWS', 'IFLYTEK'], "SA": ['ELEC', "yelp", 'SST', 'SST-5', 'PC', 'CR', 'MR', 'MPQA', 'IMDB'],
                 "TOPIC": ['R8', 'hsumed', "AG", 'sougou', 'DBPedia'], "QA": ["TREC", 'yahoo'], 'SUBJ': ['SUBJ'],
                 'ACD': ['semeval', 'sentihood'], 'STS': ['OCNLI', 'AFQMC']}


    if opt.dataset  in ['TNEWS', 'IFLYTEK']:
        opt.neg_sample_num = 10
    else:
        opt.neg_sample_num = opt.lebel_dim-1

    for k, v in task_list.items():
        if opt.dataset in v:
            opt.task = k
            break

    opt.seed= random.randint(20,300)

    if opt.dataset in ['TREC', 'SST', 'SST-5', 'CR', 'SUBJ', 'MR', 'R8']:
        opt.batch_size=16


    logger.info('dataset {}, train_sample {}'.format(opt.dataset, opt.train_sample))
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_files = {
        'labeled': '../../datasets/{1}/{0}/sub_clean.json'.format(opt.dataset, opt.task),
        'unlabeled': '../../datasets/{1}/{0}/sub_noise_new.json'.format(opt.dataset, opt.task),
        'test': '../../datasets/{1}/{0}/test.json'.format(opt.dataset, opt.task),
        'dev': '../../datasets/{1}/{0}/dev.json'.format(opt.dataset, opt.task)
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_group
    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'label']

    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.optimizer = AdamW
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.save_dir_histo = 'results/{}/'.format( opt.dataset)
    os.makedirs(opt.save_dir_histo, exist_ok=True)
    log_file = 'ssl-{}-{}.log'.format(opt.dataset, opt.noise_percentage)
    if os.path.exists(log_file): os.remove(log_file)
    logger.addHandler(logging.FileHandler(log_file))

    logger.info('seed {}'.format(opt.seed))
    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
   main()
