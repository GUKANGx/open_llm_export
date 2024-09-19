import os
from path_mngr.path_mngr import TARGET_DIR, TARGET_DUMP_DATA_DIR, RESOURCE_DIR
from model_exporter.model_exporter import ModelExporter


if __name__ == "__main__":
    print(f"support model list: ")
    for model_type in ModelExporter.support_model_type_list():
        print(f"-> model type: {model_type}")

    tiny_llama_model_type = "TinyLlama_1_1B_Chat_v1_0"
    tiny_llama_hf_model_path = os.path.join(RESOURCE_DIR, tiny_llama_model_type)
    tiny_llama_exporter = ModelExporter(seq_len=32, kv_cache_max_len=2048, is_dynamic_shape=False)
    tiny_llama_exporter.load(tiny_llama_hf_model_path, tiny_llama_model_type)
    onnx_model_path = r"E:\Project\PythonProject\open_llm_export\targets\model_export\TinyLlama_1_1B_Chat_v1_0_onnx"
    tiny_llama_exporter.export_onnx(onnx_model_path, need_test=True, prompt="hello")
