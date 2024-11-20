from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import transformers
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score

# 定义一个计算指标的函数，用于评估模型性能
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)  # 获取概率最大的预测类别
    accuracy = accuracy_score(p.label_ids, preds)  # 计算准确率
    f1 = f1_score(p.label_ids, preds, average='macro')  # 计算F1分数，使用macro平均
    return {"accuracy": accuracy, "f1": f1}

# 设置transformers日志的详细程度为信息级别
transformers.logging.set_verbosity_info()

# 加载数据集
data = pd.read_csv('clean_data.csv')  
dataset = Dataset.from_pandas(data) 

# 数据预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载BERT分词器，适用于中文数据

# 定义预处理函数，用于对数据集中的样本进行处理
def preprocess_function(examples):
    labels = [label - 1 for label in examples["Rating"]] 
    model_inputs = tokenizer(examples['Review'], truncation=True, padding=True, max_length=128)
    model_inputs["labels"] = labels  # 将处理后的标签添加到模型输入中
    return model_inputs

# 将预处理函数应用于数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 将数据集分为训练集和验证集
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# 定义模型，这里使用BERT模型进行序列分类任务，并指定标签数量
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=5)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',  
    num_train_epochs=5,  
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,  
    warmup_steps=500, 
    weight_decay=0.01,
    logging_steps=10,  
    logging_dir='./logs',  
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,  # 指定计算指标的函数
)

# 开始训练
trainer.train()
