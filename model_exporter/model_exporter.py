import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnx
import onnxsim
import onnxruntime

from . import model_impl_base
from . import model_impl_tiny_llama


class ModelExporter(object):
    def __init__(self):
        super(ModelExporter, self).__init__()
        self.tokenizer_model = None
        self.hf_model = None
        self.model_impl = None

    @staticmethod
    def support_model_type_list():
        return model_impl_base.MODEL_IMPL_FACTORY.model_type_list()

    def load(self, hf_model_pah, model_type, **kwargs):
        # 1. load tokenizer
        print(f"exporter: loading tokenizer...")
        self.tokenizer_model = AutoTokenizer.from_pretrained(hf_model_pah)
        print(f"exporter: load tokenizer, done.")
        # 2. only support hf model, other model is to supported
        # load hf model
        print(f"exporter: loading hf model...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(hf_model_pah).eval()
        print(f"exporter: load hf model, done.")
        # 3. create impl
        self.model_impl = model_impl_base.MODEL_IMPL_FACTORY.get_model(model_type)()
        self.model_impl.load(self.hf_model)

    def export_onnx(self, onnx_model_path, need_test=False, prompt="hello"):
        # 99. golden
        print(f"exporter: golden -> prompt to ids...")
        hf_model_input_tokens = self.tokenizer_model(prompt, return_tensors="pt")
        print(f"exporter: golden -> prompt to ids, done. ids: {hf_model_input_tokens}")
        print(f"exporter: golden -> generate...")
        hf_model_generate = self.hf_model.generate(hf_model_input_tokens["input_ids"],
                                                   do_sample=False,
                                                   max_length=len(hf_model_input_tokens["input_ids"]) + 8)
        print(f"exporter: golden -> generate, done. \n -> generate ids : {hf_model_generate}, "
              f"\n -> text: {self.tokenizer_model.decode(hf_model_generate.tolist()[0])}")
