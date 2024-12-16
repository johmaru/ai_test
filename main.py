import json
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import ctranslate2
import gc
import torch.nn as nn
import torch
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
    AutoConfig,
    PretrainedConfig
)
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import bitsandbytes as bnb

def prepare_dataset():

    df = pd.read_csv("ai-test-dataset.csv", na_filter=False)

    df = df.replace(r'^\s*$', None, regex=True)
    df = df.fillna("")

    label_encoders = {}
    label_columns = ["context", "emotion", "tone", "key_phrases", "chara_name", "speakstyle"]

    for col in label_columns:
        le = LabelEncoder()

        if col in ['tone', 'key_phrases']:
           unique_values = set([item.strip() for items in df[col].str.split(',') for item in items if item.strip()])
           le.fit(list(unique_values))

           df[f'{col}_encoded'] = df[col].apply(
               lambda x: [le.transform([item.strip()])[0] for item in x.split(',') if item.strip()]
               )
        else:

            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna(''))   
        label_encoders[col] = le    

    dataset = Dataset.from_pandas(df)

    return dataset, label_encoders

def create_model_inputs(tokenizer, dataset):
    def tokenize_function(examples):
        return tokenizer(
            examples["text_sample"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ['input_ids', 'attention_mask']]

        )
    return tokenized_dataset

def prepare_labels(dataset):
     label_columns = ["context", "emotion", "tone", "key_phrases", "chara_name", "speakstyle"]
     label_sizes = {}

     for col in label_columns:
         if col in ['tone', 'key_phrases']:
             label_sizes[col] = max(max(len(x) for x in dataset[f'{col}_encoded']), 1)

         else:

             label_sizes[col] = len(set(dataset[col]))

     return label_sizes       
 
class MultiLabelCharacterConfig(PretrainedConfig):
    model_type = "multilabel_character"
    
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings      
        
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
       try:
           config_path = os.path.join(pretrained_model_path, "config.json")
           
           if os.path.exists(config_path):
               with open(config_path, "r") as f:
                   config = json.load(f)
               return cls(**config)
           
           config_dict = kwargs.pop("config_dict", None)
           if config_dict is None:
            return cls(**kwargs)
           return cls(**config_dict)  
       
       except Exception as e:
           print(f"Config読み込みエラーが発生しました: {str(e)}")  
           raise


class MultiLabelCharacterModel(PreTrainedModel):
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data = module.weight.data.to(torch.float32)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data = module.weight.data.to(torch.float32)
            module.bias.data = module.bias.data.to(torch.float32)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        try:
            config = kwargs.get("config")
            if config is None:
                config = MultiLabelCharacterConfig.from_pretrained(pretrained_model_name_or_path)
            
            base_model = kwargs.get("base_model")
            if base_model is None:
                raise ValueError("base_model is required for MultiLabelCharacterModel")
            
            num_labels_dict = kwargs.get("num_labels_dict", {
                "emotion": 8,
                "tone": 4,
                "key_phrases": 10,
                "chara_name": 5,
                "speakstyle": 3
            })
            
            model = cls(config=config, base_model=base_model, num_labels_dict=num_labels_dict)
            return model
            
        except Exception as e:
            print(f"モデルのロードエラー: {str(e)}")
            raise
    
    def generate(self, *args, **kwargs):
        if hasattr(self.transformer, "generate"):
            return self.transformer.generate(*args, **kwargs)
        elif hasattr(self.base_model, "generate"):
            return self.base_model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("generate method not implemented")
    
    def __init__(self,config, base_model, num_labels_dict):
        if not base_model:
            raise ValueError("base_model cannot be None")
        if not num_labels_dict:
            raise ValueError("num_labels_dict cannot be None")
        super().__init__(config)
        self.transformer = base_model.transformer if hasattr(base_model, 'transformer') else base_model
        self.config = config

        self.classifiers = nn.ModuleDict({
            label: nn.Linear(self.config.hidden_size, num_labels).to(torch.float32)
            for label, num_labels in num_labels_dict.items()
        })
        
        for classifier in self.classifiers.values():
            for param in classifier.parameters():
                param.requires_grad = True
                param.data = param.data.to(torch.float32)
        
    def gradient_checkpointing_enable(self, **kwargs):
       if hasattr(self.transformer, "gradient_checkpointing_enable"):
              self.transformer.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids, attention_mask,labels=None, **kwargs):
        
        with torch.set_grad_enabled(self.training):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                )
            
        hidden_state = outputs.hidden_states[-1][:, 0, :]
        
        logits = {
            label: classifier(hidden_state)
            for label, classifier in self.classifiers.items()
        }
        loss = None
        if labels is not None:
            loss = torch.tensor(0.0, device=hidden_state.device, dtype=torch.float32,requires_grad=True)
            loss_fct = nn.CrossEntropyLoss()
            for label_name, label_logits in logits.items():
                if labels.get(label_name) is not None:
                    label_loss = loss_fct(label_logits.float(), labels[label_name])
                    loss = loss + label_loss
            return {"loss": loss, "logits": logits}
        
        return {"loss": torch.tensor(0.0, device=hidden_state.device, dtype=torch.float32, requires_grad=True), "logits": logits}
        

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = {k: v for k, v in inputs.items() if k in model.classifiers.keys()}
        for k in labels.keys():
            inputs.pop(k)
        
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]
        
    
def train_model():

    dataset, label_encoders = prepare_dataset()
    
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_map = "auto"
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload = True,
        llm_int8_compute_dtype=torch.float32,
    )
    

    model = "tokyotech-llm/Swallow-MS-7b-v0.1"
    base_model = AutoModelForCausalLM.from_pretrained(
        model, 
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    
    tokenized_train_dataset = create_model_inputs(tokenizer, train_dataset)
    tokenized_eval_dataset = create_model_inputs(tokenizer, eval_dataset)
    num_labels_dict = prepare_labels(dataset)
    
    model = MultiLabelCharacterModel(base_model.config,base_model, num_labels_dict)
    
    model = model.to_empty(device=device).train()

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,             
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        max_grad_norm=0.3,              
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        # optimizer設定
        optim="adamw_hf",
        warmup_ratio=0.03,              
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        dataloader_num_workers=0,        
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
    )

    trainer.train()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("saving 8bit model")
    
    trainer.save_model("./character_model_full")
    
    config = {
        "quantization": "8bit",
        "original_model": "tokyotech-llm/Swallow-MS-7b-v0.1",
        "model_type": "MultiLabelCharacterModel",
        "classifiers": list(model.classifiers.keys())
    }
    
    with open("./character_model_full/config.json", "w") as f:
        json.dump(config, f)
    
    return base_model,tokenizer

def load_trained_model():
    
    try: 
    
        model_name = "./character_model_full"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        config = AutoConfig.from_pretrained(model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload = True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            use_safetensors=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        dataset, label_encoders = prepare_dataset()
        num_labels_dict = prepare_labels(dataset)
        
        with torch.no_grad():
            model = MultiLabelCharacterModel(   
        config=base_model.config,
        base_model=base_model,
        num_labels_dict=num_labels_dict
        )
    
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    
        return model, tokenizer

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise e

def generate_response(text,model,tokenizer):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    character_name = "ユニ"
    
    prompt = f"あなたは{character_name}です。以下の質問に答えてください:\n{text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    gen_config = GenerationConfig(
       max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    output = model.generate(
         input_ids=inputs["input_ids"],
         attention_mask=inputs["attention_mask"],
         generation_config=gen_config,
     )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
        
    return response
    

def predict_character(text_sample, model, tokenizer,label_enocders):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    inputs = tokenizer(text_sample, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    decoded_predictions = {}
    for label_name, logits in outputs.items():
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=-1)
        
        if label_name in label_enocders:
            decoded_predictions[label_name] = label_enocders[label_name].inverse_transform([pred_idx.item()])[0]
    return decoded_predictions
    
if __name__ == "__main__":
    # if you want to train the model
    
   """ torch.cuda.empty_cache()
   print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

   model,tokenizer = train_model()
   
   print("Model Trained") """
   
   
   torch.cuda.empty_cache()
   model,tokenizer = load_trained_model()
   
   test_text = "好きな物は？"
   
   response = generate_response(test_text,model,tokenizer)
   print(response)
    
   
   