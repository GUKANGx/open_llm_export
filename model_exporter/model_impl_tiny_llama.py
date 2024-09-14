from torch import nn
import torch
import transformers
import onnx
import onnxsim
import onnxruntime

from . import model_impl_base


@model_impl_base.MODEL_IMPL_FACTORY.register_model("TinyLlama_1_1B_Chat_v1_0")
class TinyLlama_1_1B_Chat_v1_0Impl(model_impl_base.DecoderOnlyModelImplBase):
    def __init__(self):
        super(TinyLlama_1_1B_Chat_v1_0Impl, self).__init__()

    def load(self, hf_model, **kwargs):
        print(f"tiny llama impl: load model...")
        super(TinyLlama_1_1B_Chat_v1_0Impl, self).load(hf_model)
        print(f"tiny llama impl: load model, done.")

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                **kwargs):
        pass
