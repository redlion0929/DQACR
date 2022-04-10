from datasets import load_dataset, load_metric
from transformers import RobertaTokenizer, AlbertTokenizer
from transformers import RobertaForMultipleChoice, AlbertForMultipleChoice, Trainer, TrainingArguments
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np
from setproctitle import setproctitle
from datasets import load_from_disk

device = 'cuda' if torch.cuda.is_available() else "cpu"

model_checkpoint = 'albert-xxlarge-v2'
batch_size = 3
datasets = load_dataset('dream')

tokenizer = AlbertTokenizer.from_pretrained(model_checkpoint, use_fast = True)
model = AlbertForMultipleChoice.from_pretrained('danlou/albert-xxlarge-v2-finetuned-csqa')


encoded_datasets = load_from_disk('DQACR_dataset')

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

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


args = TrainingArguments(
    "DQACR",
    evaluation_strategy='epoch',
    learning_rate = 1e-5,
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs = 3,
    weight_decay = 0.01,
    logging_strategy = 'steps',
    logging_steps = 20,
    run_name = 'DQACR_0329',
    report_to="wandb"

)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,

)

trainer.train()

torch.save(model.state_dict(), 'DQACR.pth')

