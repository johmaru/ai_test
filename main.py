import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

os.add_dll_directory("C:\\Users\\Johma\\anaconda3\\envs\\test_ai\\Library\\bin")

import ctranslate2
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
        
        config = kwargs.pop("config", None)
        base_model = kwargs.pop("base_model", None)
        num_labels_dict = kwargs.pop("num_labels_dict", None)
        
        model = cls(config=config, base_model=base_model, num_labels_dict=num_labels_dict)
        return model
    
    def generate(self, *args, **kwargs):
        if hasattr(self.transformer, "generate"):
            return self.transformer.generate(*args, **kwargs)
        elif hasattr(self.base_model, "generate"):
            return self.base_model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("generate method not implemented")
    
    def __init__(self,config, base_model, num_labels_dict):
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
    
    """ print("Model's module names:")
    for name, _ in base_model.named_modules():
        print(name) """
    
    
    # only fine-tune the lora layers
    """  target_modules = [
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ]

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    ) """
    
   # base_model = get_peft_model(base_model, lora_config)
   # base_model.save_pretrained("./character_model_lora")
    
    model = MultiLabelCharacterModel(base_model.config,base_model, num_labels_dict)
    
    model = model.to_empty(device=device).train()
    
    
    # only fine-tune the lora layers
    """  for name,param in model.named_parameters():
        if "lora" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False """

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
        optim="adamw",
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
    
    model = model.float()
    
    model = model.to(torch.float32) 

    trainer.train()
    
    for name, param in model.named_parameters():
        if not param.dtype == torch.float32:
            param.data = param.data.to(torch.float32)

    trainer.save_model("./character_model_full")
    return base_model,tokenizer

def load_trained_model():
    
    try: 
    
        model_name = "./character_model_full"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        config = AutoConfig.from_pretrained(model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32
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
    
   torch.cuda.empty_cache()
   print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

   model,tokenizer = train_model()
   
   print("Model Trained")
   
   
   """ torch.cuda.empty_cache()
   model,tokenizer = load_trained_model()
   
   test_text = "好きな物は？"
   
   response = generate_response(test_text,model,tokenizer)
   print(response) """
    
   
   