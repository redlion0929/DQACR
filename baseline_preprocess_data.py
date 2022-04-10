from datasets import load_dataset
from transformers import RobertaTokenizer, AlbertTokenizer

model_checkpoint = 'albert-xxlarge-v2'
datasets = load_dataset('dream')
tokenizer = AlbertTokenizer.from_pretrained(model_checkpoint, use_fast = True)

def add_label(dataset):
    new_col = []
    for idx, i in enumerate(dataset):
        for idx_j,j in enumerate(i['choice']):
            if j==i['answer']:
                new_col.append(idx_j)
                break
    return dataset.add_column('label',new_col)

datasets['train'] = add_label(datasets['train'])
datasets['validation'] = add_label(datasets['validation'])
datasets['test'] = add_label(datasets['test'])

def preprocess_function(examples):
    # 선지 수만큼 dialogue를 복사
    first_sentence = [[' '.join(context)] * 3 for context in examples["dialogue"]]

    question_headers = examples['question']
    second_sentence = [[f"{header} {end}" for end in examples['choice'][i]] for i, header in
                       enumerate(question_headers)]
    first_sentences = sum(first_sentence, [])
    second_sentences = sum(second_sentence, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i:i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}

encoded_datasets = datasets.map(preprocess_function, batched = True)

encoded_datasets.save_to_disk('vanila_dataset')