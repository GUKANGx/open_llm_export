from torch import nn
import torch
import transformers
import onnx
import onnxsim
import onnxruntime


class DecoderOnlyModelImplBase(nn.Module):
    def __init__(self):
        super(DecoderOnlyModelImplBase, self).__init__()
        self.hf_model = None
        self.hf_config = None
        self.lm_head = None

    def load(self, hf_model, **kwargs):
        print(f"DecoderOnlyModelImplBase: load...")
        self.hf_model = hf_model
        self.hf_config = hf_model.config
        self.lm_head = self.hf_model.lm_head
        print(f"DecoderOnlyModelImplBase: hf_config -> {self.hf_config}")
        print(f"DecoderOnlyModelImplBase: lm_head -> {self.lm_head}")
        print(f"DecoderOnlyModelImplBase: load, done.")
        return NotImplemented

    def forward(self, **kwargs):
        return NotImplemented


class ModelImplFactory(object):
    def __init__(self):
        super(ModelImplFactory, self).__init__()
        self._model_impl_map = dict()

    def register_model(self, type_name):
        def _register(target):
            print(f"register model impl type: {type_name}, target: {target}")
            self._model_impl_map[type_name] = target
            return target
        return _register

    def get_model(self, type_name):
        return self._model_impl_map[type_name]

    def model_type_list(self):
        return list(self._model_impl_map.keys())


MODEL_IMPL_FACTORY = ModelImplFactory()
