# Welcome to MRC + CSK Pretraining Models!
need to install transformers 2.9.0 dev <br>
```
pip install git+https://github.com/huggingface/transformers.git
```

# Datasets (test)

1. MRC : MultiRC
2. Classification : CoLA(GLUE)
3. Commonsense : SocialIQA

# Baseline
- Muppet
- RoBERTa

# Todo
1. Tokenization
- [ ] Add Task specific Tokens \<cola\>, \<mrc\>, \<com\>
2. DatasetLoader
- [ ] Loads multiple tasks in one batch
- [ ] Adds task label to clarify which type of task the data belongs
3. Loss
- [ ] Scaling loss based on Muppet
- [ ] R3F/R4F Loss implementation
