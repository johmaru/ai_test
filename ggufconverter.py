import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ctranslate2.converters import TransformersConverter
from safetensors.torch import load_file

class CustomConverter(TransformersConverter):
    def __init__(self, model_name_or_path, **kwargs):
        print("CustomConverterの初期化開始...")
        try:
            # 基底クラスの初期化前に型変換を行う
            model_files = {
                'model-00001': os.path.join(model_name_or_path, 'model-00001-of-00002.safetensors'),
                'model-00002': os.path.join(model_name_or_path, 'model-00002-of-00002.safetensors')
            }
            
            print("モデルファイルをロード中...")
            state_dict = {}
            for _, file_path in model_files.items():
                if os.path.exists(file_path):
                    print(f"ロード中: {file_path}")
                    # 明示的にfloat32として読み込む
                    tensors = load_file(file_path)
                    state_dict.update({k: v.to(torch.float32) for k, v in tensors.items()})
            
            # float32の状態で基底クラスを初期化
            super().__init__(model_name_or_path)
            
            if hasattr(self, 'model'):
                self.model.load_state_dict(state_dict, strict=False)
            print("CustomConverterの初期化完了")
            
        except Exception as e:
            print(f"初期化エラー: {str(e)}")
            raise

    def convert_weights(self, *args, **kwargs):
        print("convert_weightsの開始...")
        try:
            if hasattr(self, 'model'):
                for name, param in self.model.named_parameters():
                    print(f"パラメータ: {name}, 型: {param.dtype}")
                    if param.dtype == torch.int8:
                        print(f"{name}をfloat32に変換")
                        param.data = param.data.float()
            print("convert_weightsの終了")
            return super().convert_weights(*args, **kwargs)
        except Exception as e:
            print(f"変換エラー: {str(e)}")
            raise

def main():
    try:
        model_path = "./character_model_full"
        output_path = "./character_model_gguf"
        
        print("変換プロセス開始...")
        converter = CustomConverter(model_path)
        
        print("変換処理開始...")
        converter.convert(output_path, force=True)
        print("変換完了")
        
    except Exception as e:
        print(f"エラー詳細: {str(e)}")
        print(f"エラータイプ: {type(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()