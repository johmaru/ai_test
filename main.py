import torch.nn as nn
import os
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
    def __init__(self,config, base_model, num_labels_dict):
        super().__init__(config)
        self.transformer = base_model.transformer if hasattr(base_model, 'transformer') else base_model
        self.config = config
        
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.classifiers = nn.ModuleDict({
            label: nn.Linear(self.config.hidden_size, num_labels).to(dtype=torch.bfloat16)
            for label, num_labels in num_labels_dict.items()
        })
        
        for classifier in self.classifiers.values():
          for param in classifier.parameters():
              param.requires_grad = True
        
    def gradient_checkpointing_enable(self, **kwargs):
       if hasattr(self.transformer, "gradient_checkpointing_enable"):
              self.transformer.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids, attention_mask,labels=None, **kwargs):
        
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True,use_cache=False)
            
        hidden_state = outputs.hidden_states[-1][:, 0, :]
        
        logits = {
            label: classifier(hidden_state)
            for label, classifier in self.classifiers.items()
        }
        loss = torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            for label_name, label_logits in logits.items():
                if labels.get(label_name) is not None:
                    loss += loss + loss_fct(label_logits, labels[label_name])
            
        return {"loss": loss, "logits": logits}

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
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    
    device_map = "auto"
    

    model = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2"
    base_model = AutoModelForCausalLM.from_pretrained(
        model, 
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        max_position_embeddings=2048,
        rope_scaling={"type": "linear", "factor": 2.0,"original_max_position_embeddings": 2048}
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
    
    
    # only fine-tune the lora layers
    """  for name,param in model.named_parameters():
        if "lora" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False """
            
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.float()

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5,             
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        max_grad_norm=0.3,              
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        # optimizer設定
        optim="adamw_8bit",
        warmup_ratio=0.1,              
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
        group_by_length=False,
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

    trainer.save_model("./character_model_full")
    return base_model,tokenizer

def load_trained_model():
    
    try: 
    
        model_name = "./character_model_full"
    
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=True
        )
    
     
        device_map = {
        'model.embed_tokens': 'cpu',
        'model.norm': 'cpu',
        'lm_head': 'cpu'
        }
        for i in range(30):
            device_map[f'model.layers.{i}'] = 'cpu'
        for i in range(30, 32):
            device_map[f'model.layers.{i}'] = 0
    
        base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_cache=False,
        max_memory={0: "10GB", "cpu": "32GB"}
        )
    
        # only fine-tune the lora layers
        """ target_modules = [
       "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
        ]
        lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True
        ) 
    
        base_model = PeftModel.from_pretrained(
       base_model,
       "./character_model_lora",
       is_trainable=False,
       config=lora_config,
        )"""
    
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        return base_model, tokenizer

    except Exception as e:
        print(e)
    raise



def generate_response(text,model,tokenizer):
    
    device = next(model.parameters()).device
    
    character_name = "ユニ"
    
    prompt = f"あなたは{character_name}です。以下の質問に答えてください:\n{text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        do_sample=True,
    )
    
    with torch.no_grad():
        outputs = model.generate(
                                 input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 generation_config=gen_config,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
    
   
   