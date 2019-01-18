import numpy as np

import os.path
import sys
import random
import h5py
import itertools
import re
# import tqdm
#
import pandas as pd
import scribt.data_util as data_util
import hickle as hkl
import pickle as pkl
import spacy
from utils.utils import StrToBytes

# from IPython import embed

__PATH__ = os.path.abspath(os.path.dirname(__file__))

def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

# PATHS
TGIF_DATA_DIR = os.path.normpath(os.path.join(__PATH__, '../../dataset/tgif'))
TYPE_TO_CSV = {'FrameQA': 'Train_frameqa_question.csv',
               'Count': 'Train_count_question.csv',
               'Trans': 'Train_transition_question.csv',
               'Action' : 'Train_action_question.csv'}
# assert_exists(TGIF_DATA_DIR)

VIDEO_FEATURE_DIR = os.path.join(TGIF_DATA_DIR, 'features')
# assert_exists(VIDEO_FEATURE_DIR)
eos_word = '<EOS>'

class Dataset():
    '''
        API for dataset
    '''

    def __init__(self,
                 dataset_name='train',
                 image_feature_net='resnet',
                 layer='pool5',
                 max_length=80,
                 use_moredata=False,
                 max_n_videos=None,
                 set_type=None,
                 task_type=None,
                 data_dir=None):
        self.data_dir = data_dir
        self.dataframe_dir = data_dir + 'DataFrame/'
        self.vocabulary_dir = data_dir + 'Vocabulary/'
        self.use_moredata = use_moredata
        self.dataset_name = dataset_name
        self.image_feature_net = image_feature_net
        self.layer = layer
        self.max_length = max_length
        self.max_n_videos = max_n_videos
        self.task_type = task_type
        self.set_type = set_type


    def read_df_from_csvfile(self, csv_path):
        data_path = os.path.join(self.dataframe_dir, csv_path)
        assert_exists(data_path)
        data_df = pd.read_csv(data_path, sep='\t')
        data_df = data_df.set_index('vid_id')
        data_df['row_index'] = range(1, len(data_df) + 1)  # assign csv row index
        return data_df

    def build_word_vocabulary(self, all_captions_source=None,
                              word_count_threshold=0,):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''
        # log.infov('Building word vocabulary (%s) ...', self.dataset_name)

        if all_captions_source is None:
            all_captions_source = self.get_all_captions()

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        for sentence in all_captions_source:
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        # log.info("Filtered vocab words (threshold = %d), from %d to %d",
        #          word_count_threshold, len(word_counts), len(vocab))

        # build index and vocabularies
        self.word2idx = {}
        self.idx2word = {}

        self.idx2word[0] = '.'
        self.idx2word[1] = 'UNK'
        self.word2idx['#START#'] = 0
        self.word2idx['UNK'] = 1
        for idx, w in enumerate(vocab, start=2):
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        pkl.dump(self.word2idx, open(os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.task_type), 'wb'))
        pkl.dump(self.idx2word, open(os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.task_type), 'wb'))

        word_counts['.'] = nsents
        bias_init_vector = np.array([1.0*word_counts[w] if i>1 else 0 for i, w in self.idx2word.items()])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        self.bias_init_vector = bias_init_vector

        self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,
                                                            'Total_{}_question.csv'.format(self.task_type.lower())), sep='\t')
        answers = list(set(self.total_q['answer'].values))
        answers.sort()
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(answers):
            self.ans2idx[w]=idx
            self.idx2ans[idx]=w
        pkl.dump(self.ans2idx, open(os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.task_type), 'wb'))
        pkl.dump(self.idx2ans, open(os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.task_type), 'wb'))

        # Make glove embedding.

        nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')

        max_length = len(vocab)
        GLOVE_EMBEDDING_SIZE = 300

        glove_matrix = np.zeros([max_length,GLOVE_EMBEDDING_SIZE])
        for i in range(len(vocab)):
            w = vocab[i]
            w_embed = nlp(u'%s' % w).vector
            glove_matrix[i,:] = w_embed

        vocab = pkl.dump(glove_matrix, open(os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl'%self.task_type), 'wb'))
        self.word_matrix = glove_matrix

    def load_word_vocabulary(self):

        word_matrix_path = os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl'%self.task_type)

        word2idx_path = os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.task_type)
        idx2word_path = os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.task_type)
        ans2idx_path = os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.task_type)
        idx2ans_path = os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.task_type)
        # self.build_word_vocabulary()

        with open(word_matrix_path, 'rb') as f:
            self.word_matrix = pkl.load(f)
        print("Load word_matrix from pkl file : %s" % word_matrix_path)

        with open(word2idx_path, 'rb') as f:
            self.word2idx = pkl.load(f)
        print("Load word2idx from pkl file : %s"% word2idx_path)

        with open(idx2word_path, 'rb') as f:
            self.idx2word = pkl.load(f)
        print("Load idx2word from pkl file : %s"% idx2word_path)

        with open(ans2idx_path, 'rb') as f:
            self.ans2idx = pkl.load(f)
        print("Load answer2idx from pkl file : %s"% ans2idx_path)

        with open(idx2ans_path, 'rb') as f:
            self.idx2ans = pkl.load(f)
        print("Load idx2answers from pkl file : %s"% idx2ans_path)

    def get_all_captions(self):
        '''
        Iterate caption strings associated in the vid/gifs.
        '''
        qa_data_df = pd.read_csv(os.path.join(self.dataframe_dir, TYPE_TO_CSV[self.task_type]), sep='\t')

        all_sents = []
        for row in qa_data_df.iterrows():
            all_sents.extend(self.get_captions(row))
        self.task_type
        return all_sents

    def get_captions(self, row):
        if self.task_type == 'FrameQA':
            columns = ['description', 'question', 'answer']
        elif self.task_type == 'Count':
            columns = ['question']
        elif self.task_type == 'Trans':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.task_type == 'Action':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']

        sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
        return sents

    def split_sentence_into_words(self, sentence, eos=True):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        try:
            words = data_util.clean_str(sentence).split()
        except:
            print(sentence)
            sys.exit()
        if eos:
            words = words + [eos_word]
        for w in words:
            if not w:
                continue
            yield w

    def get_train_data(self):
        assert self.task_type in ['FrameQA', 'Count', 'Trans', 'Action'], 'Should choose task type'
        data_df = self.read_df_from_csvfile(csv_path='Train_{}_question.csv'.format(self.task_type.lower()))
        test_df = self.read_df_from_csvfile(csv_path='Test_{}_question.csv'.format(self.task_type.lower()))
        self.load_word_vocabulary()
        return data_df, test_df, self.ans2idx, self.idx2ans, self.word2idx, self.idx2word, self.word_matrix


