from datasets import load_metric

TASK_METRICS = {
    "TASK1" : load_metric("accuracy"),
    "TASK2" : load_metric("glue", "cola"),
    "TASK3" : load_metric("accuracy"),
    "TASK4" : load_metric("accuracy"),
}
