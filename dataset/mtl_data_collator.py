
from dataclasses import dataclass
from typing import Optional, Union
import torch
from transformers import DataCollatorWithPadding
from transformers.file_utils import PaddingStrategy

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_sentencepiece_objects import T5Tokenizer

# Social IQA
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
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
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

CollatorFactory = {
    "CoLA": DataCollatorWithPadding,
    "MultiRC": DataCollatorWithPadding,
    "SocialIQA": DataCollatorForMultipleChoice,
    "CommonsenseQA": DataCollatorForMultipleChoice,
    "SQuAD1.1": DataCollatorWithPadding,
    "CosmosQA": DataCollatorForMultipleChoice,
}

def main():
    print("hello world!")

    
if __name__ == "__main__":
    main()