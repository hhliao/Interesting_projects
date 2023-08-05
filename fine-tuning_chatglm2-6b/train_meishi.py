# -*- coding : utf-8 -*-
# @Time     : 2023/7/20 - 11:56
# @Author   : rainbowliao
# @FileName : train_meishi.py
#
import warnings
warnings.filterwarnings("ignore")

import json
import random
import datasets
import pandas as pd
from tqdm import tqdm

import transformers

from transformers import  AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch

random.seed(42)


model_name = "THUDM/chatglm2-6b" # "../chatglm2-6b" #或者远程 “THUDM/chatglm2-6b”

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, #QLoRA 设计的 Double Quantization
    bnb_4bit_quant_type='nf4', # QLoRA 设计的 Normal Float 4 量化数据类型
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,
                                  quantization_config=bnb_config,
                                  trust_remote_code=True) #.half().cuda()

response, his = model.chat(tokenizer, '宫保鸡丁', history=[])
print(response)


def process_mstx(data_file):
    with open(data_file, 'r', encoding='utf-8-sig') as fd:
        meishi_json = json.load(fd)

    print(f"总共有{len(meishi_json)}个菜品")
    print(json.dumps(meishi_json[0], indent=2, ensure_ascii=False))

    data = []
    for food in meishi_json:
        # 食品名称
        food_name = food["title"]
        # 食材明细
        ingredient = ""
        for k, v in food["ingredients"].items():
            ingredient += f"{k} : {v} \n"
        # 制作步骤
        step = ""
        for st in food["steps"]:
            step += f"第{st['index']}步：{st['content']}"

        # 制作方法
        craft = "" if food['craft'] is None else food['craft']
        duration = "" if food['duration'] is None else food['duration']
        method = craft + duration

        # 构建数据
        data.append({
            "id": food['id'],
            "菜品名称": food_name,
            "食材明细": ingredient,
            "制作步骤": step,
            "制作方法": method,
        })
    return data

data_file = "./mstx-中文菜谱.json"
meishi_data = process_mstx(data_file)
print(meishi_data[0])

processed_data_file = 'mstx-中文菜谱-processed.json'
with open(processed_data_file, 'w', encoding='utf-8') as fd:
    fd.write(json.dumps(meishi_data, indent=4, ensure_ascii=False))

def build_foodname_methods_prompt(data_file, prompt = ""):
    '''
    给菜名，生成制作方法+步骤+食材
    '''
    with open(data_file, "r", encoding='utf8') as f:
        caipu_json = json.load(f)
    print(f'总共有{len(caipu_json)}个菜品')

    data = []
    for caiming in caipu_json:
        # print(caiming['id'])
        # 菜品名称
        food_name = caiming["菜品名称"]

        # 食材明细
        ingredient = caiming["食材明细"]

        # 制作步骤
        step = caiming["制作步骤"]

        # 制作方法
        method = caiming["制作方法"]

        # 构建prompt
        prompt_item = {'prompt' : prompt + food_name, 'response':
            '\n' + '食材明细: \n' + ingredient + '\n' + "制作步骤: \n" +  step + '\n' +
            "制作方法: \n" + method + '\n'}
        data.append(prompt_item)
    return data

def build_methods_foodname_prompt(data_file, prompt=""):
    '''
    给食材+制作方法+步骤，生成菜名
    '''
    with open(data_file, "r", encoding='utf8') as f:
        caipu_json = json.load(f)
    print(f'总共有{len(caipu_json)}个菜品')

    data = []
    for caiming in caipu_json:
        # print(caiming['id'])
        # 菜品名称
        food_name = caiming["菜品名称"]

        # 食材明细
        ingredient = caiming["食材明细"]

        # 制作步骤
        step = caiming["制作步骤"]

        # 制作方法
        method = caiming["制作方法"]

        # 构建prompt
        prompt_item = {'prompt': prompt + '\n' + '食材明细: \n' + ingredient + '\n' + "制作步骤: \n" + step + '\n' +
            "制作方法: \n" + method + '\n', 'response': "以上步骤是菜品 (" + food_name + ") 的制作方法 \n"}
        data.append(prompt_item)
    return data

def build_foodname_match_methods_prompt(
        data_file,
        neg_sample_num=2,
        prompt="文本分类任务：判断菜品与制作方式的匹配醒进行判断，分成匹配和不匹配"):
    '''
    给出 菜名 + 食材 + 制作方法 + 步骤， 给出判断是否匹配
    '''
    with open(data_file, "r", encoding='utf8') as f:
        caipu_json = json.load(f)
    total_food = len(caipu_json)
    print(f'总共有{total_food}个菜品')

    data = []
    for caiming in caipu_json:
        # print()
        id = caiming['id']
        # 菜品名称
        food_name = caiming["菜品名称"]

        # 食材明细
        ingredient = caiming["食材明细"]

        # 制作步骤
        step = caiming["制作步骤"]

        # 制作方法
        method = caiming["制作方法"]

        prompt1 = prompt + '\n' + f"烹制{food_name}的方式如下:\n" + '食材明细: \n' + ingredient + '\n' + "制作步骤: \n" + step + \
                 '\n' + "制作方法: \n" + method + '\n'
        prompt2 = prompt + '\n' + f"烹制{food_name}的方式如下:\n" + '\n' + "制作步骤: \n" + step + '食材明细: \n' + ingredient + \
                  '\n' + "制作方法: \n" + method + '\n'
        prompt3 = prompt + '\n' + f"烹制{food_name}的方式如下:\n" + '\n' + "制作步骤: \n" + step + \
                  '\n' + "制作方法: \n" + method + '食材明细: \n' + ingredient + '\n'
        prompt4 = prompt + '\n' + f"烹制{food_name}的方式如下:\n" + '食材明细: \n' + ingredient + '\n' + \
                  '\n' + "制作方法: \n" + method + "制作步骤: \n" + step + '\n'
        data.append({"prompt": prompt1, "response": "上述菜名与烹饪方式是匹配的"})
        data.append({"prompt": prompt2, "response": "上述菜名与烹饪方式是匹配的"})
        data.append({"prompt": prompt3, "response": "上述菜名与烹饪方式是匹配的"})
        data.append({"prompt": prompt4, "response": "上述菜名与烹饪方式是匹配的"})

        sample_num = 0
        while True:
            sample_id = random.randint(0, total_food-1)
            if sample_id == id:
                continue
            sample_num += 1
            if sample_num > neg_sample_num:
                break
            neg_caiming = caipu_json[sample_id]
            neg_food_name = neg_caiming["菜品名称"]
            neg_ingredient = neg_caiming["食材明细"]
            neg_step = neg_caiming["制作步骤"]
            neg_method = neg_caiming["制作方法"]

            prompt1 = prompt + '\n' + f"烹制{food_name}的方式如下:\n" + '食材明细: \n' + neg_ingredient + '\n' +\
                      "制作步骤: \n" + neg_step + '\n' + "制作方法: \n" + neg_method + '\n'
            prompt2 = prompt + '\n' + f"烹制{neg_food_name}的方式如下:\n" + '食材明细: \n' + ingredient + '\n' + "制作步骤: \n" + step + \
                      '\n' + "制作方法: \n" + method + '\n'
            data.append({"prompt": prompt1, "response": "上述菜名与烹饪方式是不匹配的"})
            data.append({"prompt": prompt2, "response": "上述菜名与烹饪方式是不匹配的"})

    return data

def split_and_comcate(data_file):
    '''
    划分训练集和测试集，按照各自数据的8：2划分
    '''
    # mathch_data = build_foodname_match_methods_prompt(data_file)
    # print(f'一共产生数据: {len(mathch_data)} 条')
    # mathch_data = pd.DataFrame(mathch_data)
    # mathch_data_ds = datasets.Dataset.from_pandas(mathch_data).train_test_split(test_size=0.2, shuffle=True, seed=42)

    food_data = build_foodname_methods_prompt(data_file)
    print(f'一共产生数据: {len(food_data)} 条')
    food_data = pd.DataFrame(food_data)
    food_data_ds = datasets.Dataset.from_pandas(food_data).train_test_split(test_size=0.2, shuffle=True, seed=42)

    method_data = build_methods_foodname_prompt(data_file)
    print(f'一共产生数据: {len(method_data)} 条')
    method_data = pd.DataFrame(method_data)
    method_data_ds = datasets.Dataset.from_pandas(method_data).train_test_split(test_size=0.2, shuffle=True, seed=42)

    # train_data = pd.concat([mathch_data_ds['train'].to_pandas(), food_data_ds['train'].to_pandas(),
    train_data = pd.concat([   food_data_ds['train'].to_pandas(), method_data_ds['train'].to_pandas()])
    test_data = pd.concat([food_data_ds['test'].to_pandas(), method_data_ds['test'].to_pandas()])
    return train_data, test_data

processed_data_file = 'mstx-中文菜谱-processed.json'
train_data, test_data = split_and_comcate(processed_data_file)
print(train_data.shape)
print(test_data.shape)
print(train_data.head(10))
train_data.to_parquet('train_data.parquet')
test_data.to_parquet('test_data.parquet')

ds_train = datasets.Dataset.from_pandas(train_data)
ds_test = datasets.Dataset.from_pandas(test_data)

#ds_train.size = 30000
#ds_test.size = 5000

max_seq_length = 1024
skip_over_length = True

# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')


def preprocess(example):
    context = example["prompt"]
    target = example["response"]

    context_ids = tokenizer.encode(
        context,
        max_length=max_seq_length,
        truncation=True)

    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)

    input_ids = context_ids + target_ids + [config.eos_token_id]

    # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss
    labels = [-100] * len(context_ids) + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids,
            "labels": labels,
            "context_len": len(context_ids),
            'target_len': len(target_ids) + 1}

ds_train_token = ds_train.map(preprocess).select_columns(['input_ids','labels', 'context_len','target_len'])
#len_ids = [len(example["input_ids"]) for example in ds_train_token]
#longest = max(len_ids)
#print(longest)
if skip_over_length:
    ds_train_token = ds_train_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)


ds_test_token = ds_test.map(preprocess).select_columns(['input_ids', 'labels','context_len','target_len'])
#len_ids = [len(example["input_ids"]) for example in ds_test_token]
#longest = max(len_ids)
#print(longest)
if skip_over_length:
    ds_val_token = ds_test_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)


def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]

        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dl_train = torch.utils.data.DataLoader(ds_train_token,num_workers=8,batch_size=8,
                                       pin_memory=True,shuffle=True, collate_fn = data_collator)
dl_test = torch.utils.data.DataLoader(ds_test_token,num_workers=8,batch_size=8,
                                    pin_memory=True,shuffle=True, collate_fn = data_collator)


from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

# model = AutoModel.from_pretrained("../chatglm2-6b",
#                                  load_in_8bit=False,  #是否导入int8量化模型
#                                  trust_remote_code=True)

model.supports_gradient_checkpointing = True  #节约cuda
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# 预处理量化模型，以适配LoRA调优
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

import bitsandbytes as bnb


def find_all_linear_modules(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(name[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


lora_modules = find_all_linear_modules(model)
print(lora_modules)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    #target_modules=lora_modules
)

model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True
model.print_trainable_parameters()

from torchkeras import KerasModel
from accelerate import Accelerator


class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        # loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"], labels=batch["labels"]).loss

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner


# 仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)


def load_ckpt(self, ckpt_path='checkpoint'):
    self.net = self.net.from_pretrained(self.net, ckpt_path)
    self.from_scratch = False


KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt

keras_model = KerasModel(model,loss_fn = None,optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6))
ckpt_path = 'meishi_chatglm2_qlora'

keras_model.fit(train_data = dl_train,
                val_data = dl_test,
                epochs=5,
                patience=3, #for early stop
                monitor='val_loss',
                mode='min',
                ckpt_path = ckpt_path,
                gradient_accumulation_steps=4,
                mixed_precision='fp16'
               )

