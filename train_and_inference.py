from datasets import load_dataset, load_metric
from transformers import RobertaTokenizer, AlbertTokenizer
from transformers import RobertaForMultipleChoice, AlbertForMultipleChoice, Trainer, TrainingArguments
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np
import argparse
from attrdict import AttrDict
import json
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.metrics import classification_report

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop('label') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

class DQACR(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model_checkpoint = 'albert-xxlarge-v2'
        self.batch_size = self.args.batch_size
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_checkpoint, use_fast = True)
        if self.args.model == 'DQACR':
            self.model = AlbertForMultipleChoice.from_pretrained('danlou/albert-xxlarge-v2-finetuned-csqa')
            self.datasets = load_from_disk('DQACR_dataset')
        else:
            self.model = AlbertForMultipleChoice.from_pretrained(self.model_checkpoint)
            self.datasets = load_from_disk('vanila_dataset')


        self.training_args = TrainingArguments(
            self.args.model,
            evaluation_strategy='epoch',
            learning_rate=self.args.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.args.num_epochs,
            weight_decay=self.args.weight_decay
        )

        self.trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(self.tokenizer),
            compute_metrics=self.compute_metrics,

        )

    def compute_metrics(self, eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    def train(self):
        self.trainer.train()
        torch.save(self.model.state_dict(), self.args.save_dirpath + self.args.load_pthpath + '.pth')

    def inference(self, model_path):
        self.test_dataset = self.datasets['test']
        self.model = AlbertForMultipleChoice.from_pretrained(self.model_checkpoint)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.preds = []
        self.y_test = []

        self.correct_question = []
        self.wrong_question = []

        for i in tqdm(range(len(self.test_dataset))):
            accepted_keys = ["input_ids", "label"]
            features = [{k: v for k, v in self.test_dataset[i].items() if k in accepted_keys}]
            X_test = DataCollatorForMultipleChoice(self.tokenizer)(features)

            outputs = self.model(**{k: v for k, v in X_test.items()})
            prediction = torch.max(outputs[1], dim=1)[1]
            answer = X_test['labels']

            self.preds.append(prediction)
            self.y_test.append(answer)

            if prediction==answer:
                self.correct_question.append(i)
            else:
                self.wrong_question.append(i)

        self.preds = list(map(int, self.preds))
        self.y_test = list(map(int, self.y_test))


        return classification_report(self.y_test, self.preds, digits=4)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="DQACR")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--config_file", dest="config_file", type=str, default="config",
                            help="Config json file")
    parsed_args = arg_parser.parse_args()

    # Read from config file and make args
    with open("{}.json".format(parsed_args.config_file)) as f:
        args = AttrDict(json.load(f))

    # print("Training/evaluation parameters {}".format(args))
    dqacr= DQACR(args)
    if parsed_args.mode == 'train':
        dqacr.train()
    else:
        display_num = 20
        result = dqacr.inference(args.save_dirpath + args.load_pthpath + '.pth')
        print("Classification Result")
        print(result[0])
        print("--------------------------------------------------------------")
        print("Correct Question")
        print("개수 : ", len(dqacr.correct_question))
        print(dqacr.correct_question[:display_num])
        print("--------------------------------------------------------------")
        print("Wrong Question")
        print("개수 : ", len(dqacr.wrong_question))
        print(dqacr.wrong_question[:display_num])

