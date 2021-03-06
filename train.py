import argparse
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from pathtracing.Models import PathTracing
from pathtracing.Optim import ScheduledOptim

import load_data as ld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def cal_performance(pred, gold, data_val_ofpa=None, smoothing=False, len=32, batch_size=128):

    loss = cal_loss(pred, gold, smoothing)
    acc_50 = 0
    acc_75 = 0
    acc_90 = 0
    acc_100 = 0
    #
    pred = pred.max(1)[1]
    reshape_gold_50 = gold.view(-1, len)[:, :int(len * 0.5)]
    reshape_gold_75 = gold.view(-1, len)[:, :int(len * 0.75)]
    reshape_gold_100 = gold.view(-1, len)[:, :int(len)]
    reshape_gold_90 = gold.view(-1, len)[:, :int(len * 0.9)]
    reshape_pred_50 = pred.view(-1, len)[:, :int(len * 0.5)]
    reshape_pred_75 = pred.view(-1, len)[:, :int(len * 0.75)]
    reshape_pred_100 = pred.view(-1, len)[:, :int(len)]
    reshape_pred_90 = pred.view(-1, len)[:, :int(len * 0.9)]
    gold = gold.contiguous().view(-1)
    for i in range(batch_size):
        if reshape_pred_50[i].equal(reshape_gold_50[i]):
            acc_50 += 1
        if reshape_gold_75[i].equal(reshape_pred_75[i]):
            acc_75 += 1
        if reshape_gold_90[i].equal(reshape_pred_90[i]):
            acc_90 += 1
        if reshape_gold_100[i].equal(reshape_pred_100[i]):
            acc_100 += 1
    non_pad_mask = gold.ne(0)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    accrl = [acc_50, acc_75, acc_90, acc_100]
    return loss, n_correct, accrl


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing, opt):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_token_total = 0
    n_token_correct = 0
    step = 1
    n_post_correct = 0
    n_ofpa_all = 0
    n_ofpa_correct = 0
    accra = [0, 0, 0, 0]

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        src_seq, tgt_seq = map(lambda x: x.to(device).to(torch.int64), batch)
        #
        src_pos = torch.tensor(get_position(src_seq.shape)).to(device)
        tgt_pos = torch.tensor(get_position(tgt_seq.shape)).to(device)
        # gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct, accrl = cal_performance(pred, tgt_seq, smoothing=smoothing, len=opt.batch_y,
                                                 batch_size=opt.batch_size)

        # print(loss)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()
        # if (step % 50) == 0:
        #     writer.add_scalar('ACT-Train/loss/50step', loss.item(), step)
        #     writer.close()

        non_pad_mask = tgt_seq.ne(0)
        n_token = non_pad_mask.sum().item()
        n_token_total += n_token
        n_token_correct += n_correct

        for i in range(4):
            accra[i] += accrl[i]

    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total

    accra[0] = accra[0] / (n_token_total /opt.batch_y)
    accra[1] = accra[1] / (n_token_total /opt.batch_y)
    accra[2] = accra[2] / (n_token_total /opt.batch_y)
    accra[3] = accra[3] / (n_token_total /opt.batch_y)

    return loss_per_token, accuracy, accra


def get_position(shape):
    pos = []
    pos_i = []
    for i in range(shape[1]):
        pos_i.append(i + 1)
    for i in range(shape[0]):
        pos.append(pos_i)
    return pos


def eval_epoch(model, validation_data, device, data_val_ofpa, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_token_total = 0
    n_token_correct = 0
    accra = [0, 0, 0, 0]
    step = 1
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validating) ', leave=False):
            # prepare data
            src_seq, tgt_seq = map(lambda x: x.to(device).to(torch.int64), batch)
            # gold = tgt_seq[:, 1:]
            src_pos = torch.tensor(get_position(src_seq.shape)).to(device)
            tgt_pos = torch.tensor(get_position(tgt_seq.shape)).to(device)
            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct, accrl = cal_performance(pred, tgt_seq, data_val_ofpa, smoothing=False,len=opt.batch_y,batch_size=opt.batch_size)

            # note keeping
            total_loss += loss.item()
            non_pad_mask = tgt_seq.ne(0)
            n_token = non_pad_mask.sum().item()
            n_token_total += n_token
            n_token_correct += n_correct
            step += 1
            for i in range(4):
                accra[i] += accrl[i]

    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total
    accra[0] = accra[0] / (n_token_total /opt.batch_y)
    accra[1] = accra[1] / (n_token_total /opt.batch_y)
    accra[2] = accra[2] / (n_token_total /opt.batch_y)
    accra[3] = accra[3] / (n_token_total /opt.batch_y)
    return loss_per_token, accuracy, accra


def train(model, training_data, validation_data, optimizer, device, opt, data_val_ofpa):
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + str(time.clock()) + '.train.log'
        log_valid_file = opt.log + str(time.clock()) + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write(str(opt) + '\n')
            log_vf.write(str(opt) + '\n')
    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, accra = train_epoch(model, training_data, optimizer, device,
                                                    smoothing=opt.label_smoothing, opt=opt)
        t_elapse = (time.time() - start) / 60
        print('  - (train epoch) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'acc50: {acc50:3.3f} %,acc75: {acc75:3.3f} %,acc90: {acc90:3.3f} %,acc100: {acc100:3.3f} %,'
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, acc50=100 * accra[0], acc75=100 * accra[1], acc90=100 * accra[2],
            acc100=100 * accra[3],
            elapse=t_elapse))
        start = time.time()
        valid_loss, valid_accu, accra = eval_epoch(model, validation_data, device, data_val_ofpa, opt=opt)
        print('  - (Validation epoch) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'acc50: {acc50:3.3f} %,acc75: {acc75:3.3f} %,acc90: {acc90:3.3f} %,acc100: {acc100:3.3f} %,'
              'elapse: {elapse:3.3f} min'.format(
            loss=valid_loss, accu=100 * valid_accu, acc50=100 * accra[0], acc75=100 * accra[1], acc90=100 * accra[2],
            acc100=100 * accra[3],
            elapse=(time.time() - start) / 60))
        valid_accus += [valid_accu]
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_all', default='data/csv/data_train_2_sort.torch')
    parser.add_argument('-save_model', default='module/pathtracing.pt')
    parser.add_argument('-start_time', default='2018-07-01')
    parser.add_argument('-end_time', default='2018-12-01')
    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-embedding_dim', type=int, default=1024)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=32)
    parser.add_argument('-d_v', type=int, default=32)
    parser.add_argument('-dimension', type=int, default=8)
    parser.add_argument('-L_layers', type=int, default=2)
    parser.add_argument('-warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-log', default='log/logs.log')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-batch_x', default=32)
    parser.add_argument('-batch_y', default=32)
    parser.add_argument('-train_type', default='name')

    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    opt.d_token_vec = opt.embedding_dim
    training_data, validation_data, voc_name, data_val_ofpa = ld.get_data_loader(opt, device)
    opt.src_vocab_size = voc_name
    opt.tgt_vocab_size = opt.src_vocab_size
    print(opt)

    pathtracing = PathTracing(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.batch_x,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        embedding_dim=opt.embedding_dim,
        d_token_vec=opt.d_token_vec,
        d_inner=opt.d_inner_hid,
        L_layers=opt.L_layers,
        dimension=opt.dimension,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, pathtracing.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.embedding_dim, opt.warmup_steps)
    if opt.train_type == 'time':
        print("train time dim ")
    else:
        train(pathtracing, training_data, validation_data, optimizer, device, opt, data_val_ofpa)


if __name__ == '__main__':
    main()
