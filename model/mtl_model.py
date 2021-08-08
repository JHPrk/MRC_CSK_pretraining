import torch
import torch.nn as nn
import transformers
import logging
logging.basicConfig(level=logging.INFO)

MODEL_TYPE_DICT={
        "span": transformers.AutoModelForQuestionAnswering,
        "cls1": transformers.AutoModelForSequenceClassification,
        "cls2": transformers.AutoModelForSequenceClassification,
        "mc": transformers.AutoModelForMultipleChoice,
    }
MODEL_NUM_LABEL_DICT={
        "span": 50800,
        "cls1": 2,
        "cls2": 2,
        "mc": 5,
    }
TASK_TYPE_AND_MODEL_TYPE={
    "TASK1" : "cls1",
    "TASK2" : "cls2",
    "TASK3" : "mc",
    "TASK4" : "mc",
    "TASK5" : "span"
}
class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_types):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name in model_types:
            model = MODEL_TYPE_DICT[task_name].from_pretrained(
                model_name, 
                config=transformers.AutoConfig.from_pretrained(model_name, num_labels=MODEL_NUM_LABEL_DICT[task_name]),
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, **kwargs):
        results = {}
        #return 0
        for key in kwargs:
            task_model_key = TASK_TYPE_AND_MODEL_TYPE[key]
            logits = self.taskmodels_dict[task_model_key](**kwargs[key])
            results[key] = logits

        return results
"""
model_name = "roberta-base"
multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict=["span", "cls", "mc"]
)

if model_name.startswith("roberta-"):
    print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["span"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["cls"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["mc"].roberta.embeddings.word_embeddings.weight.data_ptr())
else:
    print("Exercise for the reader: add a check for other model architectures =)")
"""