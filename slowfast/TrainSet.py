from __future__ import absolute_import
import os
import numpy as np
import re
import h5py
from PIL import Image
import torch
from torchvision.transforms import transforms

from scribt.data_util import pad_video, pad_sequences


class TrainSet(object):
    def __init__(self, dataset, hdf5_path, word_matrix, word2idx, ans2idx):
        #, questions, word_embed,labels,root=None, transform=None):
        super(TrainSet, self).__init__()
        self.dataset = dataset
        self.feat_h5 = h5py.File(hdf5_path, 'r')
        self.image_feature_net = 'resnet'
        self.layer = 'pool5'
        self.word_matrix = word_matrix
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            data_list = []
            for index in indices:
                data_list.extend(self.get_single_item(index))
            return data_list
        return self.get_single_item(indices)

    def get_single_item(self, index):
        text_data = self.dataset[index]
        img_path, question, answer, type, video_id, caption, _ = text_data
        # video_feature = self.load_video_feature(video_id)
        # video_feature = pad_video(video_feature, [35,2048])
        imgs = self.load_imgs(img_path)
        question_embed, ql = self.convert_sentence_to_matrix(question)
        label = self.ans2idx[answer]
        return imgs, question_embed, ql, label, type

    def load_video_feature(self, video_id):

        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['pool5', 'res5c']
            video_feature = np.array(self.feat_h5[str(video_id)])
            return  video_feature
            # if self.layer.lower() == 'res5c':
            #     video_feature = np.transpose(
            #         video_feature.reshape([-1, 2048, 7, 7]), [0, 2, 3, 1])
            #     assert list(video_feature.shape[1:]) == [7, 7, 2048]
            # elif self.layer.lower() == 'pool5':
            #     video_feature = np.expand_dims(video_feature, axis=1)
            #     video_feature = np.expand_dims(video_feature, axis=1)
            #     assert list(video_feature.shape[1:]) == [1, 1, 2048]
        # elif self.image_feature_net.lower() == 'c3d':
        #     assert self.layer.lower() in ['fc6', 'conv5b']
        #     video_feature = np.array(self.feat_h5[video_id])
        #
        #     if self.layer.lower() == 'fc6':
        #         if len(video_feature.shape) == 1:
        #             video_feature = np.expand_dims(video_feature, axis=0)
        #         video_feature = np.expand_dims(video_feature, axis=1)
        #         video_feature = np.expand_dims(video_feature, axis=1)
        #         assert list(video_feature.shape[1:]) == [1, 1, 4096]
        #     elif self.layer.lower() == 'conv5b':
        #         if len(video_feature.shape) == 4:
        #             video_feature = np.expand_dims(video_feature, axis=0)
        #         video_feature = np.transpose(
        #             video_feature.reshape([-1, 1024, 7, 7]), [0, 2, 3, 1])
        #         assert list(video_feature.shape[1:]) == [7, 7, 1024]
        #
        # elif self.image_feature_net.lower() == 'concat':
        #     assert self.layer.lower() in ['fc', 'conv']
        #     c3d_feature = np.array(self.feat_h5["c3d"][video_id])
        #     resnet_feature = np.array(self.feat_h5["resnet"][video_id])
        #     if len(c3d_feature.shape) == 1:
        #         c3d_feature = np.expand_dims(c3d_feature, axis=0)
        #     # if len(resnet_feature.shape) == 1:
        #     #    resnet_feature = np.expand_dims(resnet_feature, axis=0)
        #
        #     if not len(c3d_feature) == len(resnet_feature):
        #         max_len = min(len(c3d_feature), len(resnet_feature))
        #         c3d_feature = c3d_feature[:max_len]
        #         resnet_feature = resnet_feature[:max_len]
        #
        #     if self.layer.lower() == 'fc':
        #         video_feature = np.concatenate((c3d_feature, resnet_feature),
        #                                        axis=len(c3d_feature.shape) - 1)
        #         video_feature = np.expand_dims(video_feature, axis=1)
        #         video_feature = np.expand_dims(video_feature, axis=1)
        #         assert list(video_feature.shape[1:]) == [1, 1, 4096 + 2048]
        #     elif self.layer.lower() == 'conv':
        #         c3d_feature = np.transpose(c3d_feature.reshape([-1, 1024, 7, 7]), [0, 2, 3, 1])
        #         resnet_feature = np.transpose(resnet_feature.reshape([-1, 2048, 7, 7]), [0, 2, 3, 1])
        #         video_feature = np.concatenate((c3d_feature, resnet_feature),
        #                                        axis=len(c3d_feature.shape) - 1)
        #         assert list(video_feature.shape[1:]) == [7, 7, 1024 + 2048]

        # return video_feature

    def load_imgs(self, img_path):
        img_root = '/home/yuan/project/tgif-qa/code/dataset/tgif/frames'
        path = os.path.join(img_root, img_path) + '.gi'
        img_list = os.listdir(path)
        img_list = sorted(img_list, key=lambda x:int(x.split('.')[0]))
        img_list = self.pad_imgs(img_list=img_list)
        imgs = []
        for i in img_list:
            ip = os.path.join(path, i)
            img = Image.open(ip).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 1)
        return imgs

    def pad_imgs(self, img_list, max_length=64):
        '''
        Fill pad to imgs to have same length.
        Pad in Left.
        '''
        length = len(img_list)
        num_padding = length - max_length

        if num_padding == 0:
            padded_imgs = img_list
        else:
            steps = np.linspace(0, length, num=max_length, endpoint=False, dtype=np.int32)
            img_list = np.array(img_list)
            padded_imgs = list(img_list[steps])
        return padded_imgs

    def convert_sentence_to_matrix(self, sentence):
        words = re.split('[ \'-,]', sentence.strip('\ \?\.\n'))
        words_pad = pad_sequences(words, max_length=16)
        sent2indices = [self.word2idx[w] if w in self.word2idx else 2 for w in words_pad]
        word_embeds = [np.float32(self.word_matrix[x-2]) for x in sent2indices]
        return word_embeds, len(words)







