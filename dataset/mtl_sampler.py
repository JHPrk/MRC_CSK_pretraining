
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset.mtl_data_collator import CollatorFactory
from torch.utils.data import DataLoader, RandomSampler

# datasampler (Distributed 이용 필요) or parallel
def compute_sampling_probability(task_configs, data_type, task_ids):
    probability = []
    total_size = 0
    for task_id in task_ids:
        dataset_size = task_configs[task_id].datasets[data_type].num_rows
        probability.append(dataset_size)
        total_size += dataset_size
    probability[:] = [ x / total_size for x in probability]
    return probability, total_size

SamplerFactory = {
    "CoLA": RandomSampler,
    "MultiRC": RandomSampler,
    "SocialIQA": RandomSampler,
    "CommonsenseQA": RandomSampler,
    "SQuAD1.1": RandomSampler,
    "CosmosQA": RandomSampler
}

def make_dataset_loader(dataset):
    print("hello world!")

def main():
    print("hello world!")

if __name__ == "__main__":
    main()
## 데이터 셋 개수로 Sampling하기 todo##

