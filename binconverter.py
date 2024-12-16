import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from ctranslate2.converters import TransformersConverter
from safetensors.torch import load_file
from main import MultiLabelCharacterModel, MultiLabelTrainer,MultiLabelCharacterConfig

class CustomConverter(TransformersConverter):
    def __init__(self, model_name_or_path, **kwargs):
        print("CustomConverterの初期化開始...")
        try: 
            
            base_model = AutoModelForCausalLM.from_pretrained("tokyotech-llm/Swallow-MS-7b-v0.1",
                                                              device_map="cpu",
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.float16,
                                                              )
            print("ベースモデルをロード中...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            print("ファインチューニング済みの重みをロード中...")
            state_dict = {}
            for i in range(1,3):
                file_path = os.path.join(model_name_or_path, f"model-{i:05d}-of-00002.safetensors")
                if os.path.exists(file_path):
                    print(f"重みファイルをロード中...: {file_path}")
                    partial_state_dict = load_file(file_path)
                    state_dict.update(partial_state_dict)
            
            base_model.load_state_dict(state_dict,strict=False)
            print("ファインチューニング済みの重みをロード完了")
            
            temp_path = "./temp_model"
            os.makedirs(temp_path, exist_ok=True)
        
            
            base_model.save_pretrained(temp_path,max_shard_size="500MB", safe_serialization=True)
            tokenizer.save_pretrained(temp_path)
            
            super().__init__(temp_path)
            self.model = base_model
            
            print("モデルファイルをロード中...")
            
            print("CustomConverterの初期化完了")
            
        except Exception as e:
            print(f"初期化エラー: {str(e)}")
            raise
        
    def __del__(self):
        import shutil
        if os.path.exists("./temp_model"):
            shutil.rmtree("./temp_model")    

    def convert_weights(self, *args, **kwargs):
        print("convert_weightsの開始...")
        try:    
           
            kwargs['quantization'] = "int8" 
            return super().convert_weights(*args, **kwargs)
        except Exception as e:
            print(f"変換エラー: {str(e)}")
            raise

def main():
    try:
        model_path = "./character_model_full"
        output_path = "./character_model_bin"
        
        print("変換プロセス開始...")
        converter = CustomConverter(model_path)
        
        print("変換処理開始...")
        converter.convert(output_path, force=True,quantization='int8')
        print("変換完了")
        
    except Exception as e:
        print(f"エラー詳細: {str(e)}")
        print(f"エラータイプ: {type(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()