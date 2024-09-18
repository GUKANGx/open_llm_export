from torch import nn
import torch
import transformers
import onnx
import onnxsim
import onnxruntime

from . import model_base


TinyLlama_1_1B_Chat_v1_0_type = "TinyLlama_1_1B_Chat_v1_0"


@model_base.MODEL_IMPL_FACTORY.register_model(TinyLlama_1_1B_Chat_v1_0_type)
class TinyLlama_1_1B_Chat_v1_0Impl(model_base.DecoderOnlyModelImplBase):
    def __init__(self):
        super(TinyLlama_1_1B_Chat_v1_0Impl, self).__init__()

    def load(self, hf_model, **kwargs):
        print(f"tiny llama impl: load model...")
        super(TinyLlama_1_1B_Chat_v1_0Impl, self).load(hf_model, **kwargs)
        print(f"tiny llama impl: load model, done.")

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                **kwargs):
        print(f"tiny llama impl: forward model...")
        model_output = super(TinyLlama_1_1B_Chat_v1_0Impl, self).forward(input_ids=input_ids,
                                                                         attention_mask=attention_mask,
                                                                         position_ids=position_ids,
                                                                         past_key_values=past_key_values,
                                                                         inputs_embeds=inputs_embeds,
                                                                         **kwargs)
        print(f"tiny llama impl: forward model, done.")
        return model_output


@model_base.MODEL_HF_CREATOR_FACTORY.register_hf_creator(TinyLlama_1_1B_Chat_v1_0_type)
class TinyLlama_1_1B_Chat_v1_0HfCreator(model_base.ModelHfCreatorBase):
    def __init__(self):
        super(TinyLlama_1_1B_Chat_v1_0HfCreator, self).__init__()
    
    def get_hf_model(self, hf_model_path):
        return super(TinyLlama_1_1B_Chat_v1_0HfCreator, self).get_hf_model(hf_model_path)

    def get_tokenizer(self, tokenizer_path):
        return super(TinyLlama_1_1B_Chat_v1_0HfCreator, self).get_tokenizer(tokenizer_path)


@model_base.MODEL_EXPORT_HELPER_FACTORY.register_export_helper(TinyLlama_1_1B_Chat_v1_0_type)
class TinyLlama_1_1B_Chat_v1_0ExportHelper(model_base.ModelExportHelperBase):
    def __init__(self, hf_config=None, seq_len=None, kv_cache_max_len=None):
        super(TinyLlama_1_1B_Chat_v1_0ExportHelper, self).__init__(hf_config=hf_config,
                                                                   seq_len=seq_len,
                                                                   kv_cache_max_len=kv_cache_max_len)
    
    def get_model_example_inputs(self):
        return super(TinyLlama_1_1B_Chat_v1_0ExportHelper, self).get_model_example_inputs()

