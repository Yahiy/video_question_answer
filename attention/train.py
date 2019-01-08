import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os.path as osp
import os
import numpy as np
from operator import itemgetter
from scribt.dataset import Dataset
from utils.TrainSet import TrainSet
from attention.attention_model import TgifModel
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint, pack_paded_questions

from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0,'


def test(args, data, test_csv, hdf5):
    data_all = test_csv.values
    test_len = data_all.shape[0]
    test_set = TrainSet(data_all, hdf5, data.word_matrix, data.word2idx, data.ans2idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = TgifModel()
    model = nn.DataParallel(model).cuda()
    checkpoint = load_checkpoint(osp.join(args.model_dir, 'checkpoint_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    acc_all = 0
    for j, d in enumerate(test_loader):
        video_features, question_embeds, ql, ans_labels = d
        imgs = Variable(video_features.cuda())
        # question_embeds, ql = pack_paded_questions(question_embeds, ql)
        question_embeds = torch.stack(question_embeds, 1).cuda()
        questions_embed = Variable(question_embeds)
        ans_labels = Variable(ans_labels.cuda())
        ans_scores = model(imgs, questions_embed, ql)
        _, preds = torch.max(ans_scores, 1)
        acc = torch.sum((preds == ans_labels).data)
        acc_all += acc
        if j % args.print_freq == 0:
            print('test img {} acc is : {}'.format(j, acc))
    print('test acc is : {:06f}'.format(int(acc_all)/int(test_len)))
    return acc_all


def valid(args, val_loader):
    model = TgifModel()
    model = nn.DataParallel(model).cuda()
    checkpoint = load_checkpoint(osp.join(args.model_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    acc_all = 0
    for j, d in enumerate(val_loader):
        video_features, question_embeds, ql, ans_labels = d
        imgs = Variable(video_features.cuda())
        # question_embeds, ql = pack_paded_questions(question_embeds, ql)
        question_embeds = torch.stack(question_embeds, 1).cuda()
        questions_embed = Variable(question_embeds)
        ans_labels = Variable(ans_labels.cuda())
        ans_scores = model(imgs, questions_embed, ql)
        _, preds = torch.max(ans_scores, 1)
        acc = torch.sum((preds == ans_labels).data)
        acc_all += acc
        if j % args.print_freq == 0:
            print('val img {} acc is : {:06d}'.format(j,acc))
    return acc_all


def main(args):
    data = Dataset(task_type='FrameQA', data_dir='/home/stage/yuan/tgif-qa/code/dataset/tgif/')
    hdf5 = '/home/stage/yuan/tgif-qa/code/dataset/tgif/features/TGIF_RESNET_pool5.hdf5'
    data_csv, test_df, ans2idx, idx2ans, word2idx, idx2word, word_matrix = data.get_train_data()
    data_all = data_csv.values
    np.random.shuffle(data_all)
    val_data = data_all[:7392]
    train_data = data_all[7392:]
    val_len = val_data.shape[0]

    train_set = TrainSet(train_data, hdf5, data.word_matrix, data.word2idx, data.ans2idx)
    val_set = TrainSet(val_data, hdf5, data.word_matrix, data.word2idx, data.ans2idx)
    train_loader = DataLoader(train_set,batch_size=args.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set,batch_size=args.batch_size, shuffle=False, pin_memory=True)

    test(args, data, test_df, hdf5)
    model = TgifModel()
    print(model)
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best = 0
    for epoch in range(0, args.epochs):
        model.train()
        losses = AverageMeter()
        corrects = AverageMeter()
        for i, d in enumerate(train_loader):
            video_features, question_embeds, ql, ans_labels = d
            imgs = Variable(video_features.cuda())
            # question_embeds, ql = pack_paded_questions(question_embeds, ql)
            question_embeds = torch.stack(question_embeds, 1).cuda()
            questions_embed = Variable(question_embeds)
            ans_labels = Variable(ans_labels.cuda())
            ans_scores = model(imgs, questions_embed, ql)
            _, preds = torch.max(ans_scores, 1)
            loss = criterion(ans_scores, ans_labels)

            losses.update(loss.data[0], ans_labels.size(0))
            corrects.update(torch.sum((preds == ans_labels).data))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t Loss {:.6f} ({:.6f})\t acc {} ({})\t'
                      .format(epoch, i + 1, len(train_loader),
                              losses.val, losses.avg, corrects.val, corrects.avg))
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint.pth.tar'))
        acc = valid(args, val_loader)
        print('valid acc {:.6f}'.format(int(acc)/int(val_len)))
        if acc > best:
            best = acc
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': 0,
            }, False, fpath=osp.join(args.model_dir, 'checkpoint_best.pth.tar'))
            print('save model best at ep {}'.format(epoch))

    test(args, data, test_df, hdf5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--print-freq', type=int, default=50)

    parser.add_argument('--model-dir', type=str, metavar='PATH', default='/home/stage/yuan/Video_QA/')

    main(parser.parse_args())