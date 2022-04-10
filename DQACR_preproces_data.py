from datasets import load_dataset, load_metric
from transformers import RobertaTokenizer, AlbertTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np
from setproctitle import setproctitle
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm


class preprocess_dataset(object):
    def __init__(self, type):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model_checkpoint = 'albert-xxlarge-v2'
        self.datasets = load_dataset('dream')
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_checkpoint, use_fast = True)
        self.embedder = SentenceTransformer('all-mpnet-base-v2')

        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD MODEL DONE
            # -------------------------------------------------------------------------
            """
        )

        self.datasets['train'] = self.add_label(self.datasets['train'])
        self.datasets['validation'] = self.add_label(self.datasets['validation'])
        self.datasets['test'] = self.add_label(self.datasets['test'])

        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD DATASET DONE
            # -------------------------------------------------------------------------
            """
        )

        if type == 'DQACR':
            print(
                """
                # -------------------------------------------------------------------------
                #   DQACR PREPROCESS START
                # -------------------------------------------------------------------------
                """
            )
            # preprocess conceptnet
            self.concept_data = pd.read_csv('conceptnet5_eng.csv')
            rel_set = self.concept_data['rel']
            arg1_set = self.concept_data['arg1']
            arg2_set = self.concept_data['arg2']

            rel = list(map(lambda x: x.split('/')[2], rel_set))
            arg1 = list(map(lambda x: x.split('/')[3], arg1_set))
            arg2 = list(map(lambda x: x.split('/')[3], arg2_set))

            triple_sent = [arg1[i] + ' ' + rel[i] + ' ' + arg2[i] for i in range(len(rel))]

            self.triple_embeddings = self.embedder.encode(triple_sent, convert_to_tensor=True)

            print(
                """
                # -------------------------------------------------------------------------
                #   TRIPLE EMBEDDING DONE
                # -------------------------------------------------------------------------
                """
            )

            self.datasets['train'] = self.add_triple_choice(self.datasets['train'])
            self.datasets['validation'] = self.add_triple_choice(self.datasets['validation'])
            self.datasets['test'] = self.add_triple_choice(self.datasets['test'])

            print(
                """
                # -------------------------------------------------------------------------
                #   ADD TRIPLE DONE
                # -------------------------------------------------------------------------
                """
            )

            dialogue_train = self.semantic_dialogue_npair(self.datasets['train'])
            self.datasets['train'] = self.datasets['train'].remove_columns('dialogue')
            self.datasets['train'] = self.datasets['train'].add_column('dialogue', dialogue_train)

            dialogue_eval = self.semantic_dialogue_npair(self.datasets['validation'])
            self.datasets['validation'] = self.datasets['validation'].remove_columns('dialogue')
            self.datasets['validation'] = self.datasets['validation'].add_column('dialogue', dialogue_eval)

            dialogue_test = self.semantic_dialogue_npair(self.datasets['test'])
            self.datasets['test'] = self.datasets['test'].remove_columns('dialogue')
            self.datasets['test'] = self.datasets['test'].add_column('dialogue', dialogue_test)

            print(
                """
                # -------------------------------------------------------------------------
                #   DQACR PREPROCESS DONE
                # -------------------------------------------------------------------------
                """
            )

        self.encoded_datasets = self.datasets.map(self.preprocess_function, batched=True)

        if type=='DQACR':
            self.encoded_datasets.save_to_disk('DQACR_dataset')
        else:
            self.encoded_datasets.save_to_disk('vanila_dataset')

        print(
            """
            # -------------------------------------------------------------------------
            #   PREPROCESS DONE
            # -------------------------------------------------------------------------
            """
        )

    def add_label(self):
        new_col = []
        for idx, i in enumerate(self.datasets):
            for idx_j, j in enumerate(i['choice']):
                if j == i['answer']:
                    new_col.append(idx_j)
                    break
        return self.datasets.add_column('label', new_col)

    def add_triple_choice(self, dataset):
        new_col = []
        for i in tqdm(dataset):
            new_triple = []
            for j in range(3):
                # semantic search in triple
                query_embedding = self.embedder.encode(i['choice'][j], convert_to_tensor=True)

                cos_scores_triple = util.cos_sim(query_embedding, self.triple_embeddings)[0]
                top_results_triple_idx = torch.topk(cos_scores_triple, k=1)[1][0]

                new_triple.append(self.triple_sent[top_results_triple_idx])
            new_col.append(new_triple)
        return dataset.add_column('triple_choice', new_col)

    def semantic_dialogue_npair(self,dataset):
        new_col = []
        for c_idx, corpus in enumerate(dataset['dialogue']):

            corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)
            query_embedding = self.embedder.encode(dataset['question'][c_idx], convert_to_tensor=True)

            # semantic search in corpus
            top_k = len(corpus)

            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            total_len = 1000
            last_idx = 0
            while total_len > 512:
                new_dialogue = []
                if last_idx == 0:
                    for idx in sorted(top_results[1][:]):
                        new_dialogue.append(corpus[idx])
                else:
                    for idx in sorted(top_results[1][:last_idx]):
                        new_dialogue.append(corpus[idx])

                max_c = 0
                for c in dataset['choice'][c_idx]:
                    input_ids_c = len(self.tokenizer(c)['input_ids'])
                    max_c = max(input_ids_c, max_c)

                max_t = 0
                for c in dataset['triple_choice'][c_idx]:
                    input_ids_c = len(self.tokenizer(c)['input_ids'])
                    max_t = max(input_ids_c, max_t)

                total_len = len(self.tokenizer(' '.join(new_dialogue))['input_ids']) \
                            + len(self.tokenizer(dataset['question'][c_idx])['input_ids']) \
                            + max_c +max_t - 4

                last_idx = last_idx - 1

            new_col.append(new_dialogue)
        return new_col

    def preprocess_function(self, examples):
        # 선지 수만큼 dialogue를 복사
        #triple_sentence = [[triple] * 3 for triple in examples["triple"]]

        first_sentence = []
        for idx, context in enumerate(examples["dialogue"]):
            first_sentence.append([examples['triple_choice'][idx][0] + '[SEP]'+ (' '.join(context)),
                               examples['triple_choice'][idx][1] + '[SEP]'+ (' '.join(context)),
                               examples['triple_choice'][idx][2] + '[SEP]'+ (' '.join(context))])

        question_headers = examples['question']
        second_sentence = [[f"{header} {end}" for end in examples['choice'][i]] for i, header in
                           enumerate(question_headers)]

        first_sentences = sum(first_sentence, [])
        second_sentences = sum(second_sentence, [])

        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)
        return {k: [v[i:i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}




