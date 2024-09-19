from torch import nn
import torch
import transformers
import onnx
import onnxsim
import onnxruntime
from transformers import AutoTokenizer, AutoModelForCausalLM


class DecoderOnlyModelImplBase(nn.Module):
    def __init__(self):
        super(DecoderOnlyModelImplBase, self).__init__()
        # default for llama
        self.hf_model = None
        self.hf_config = None
        self.embedding = None
        self.decode_layers = None
        self.norm = None
        self.lm_head = None

    def load(self, hf_model, **kwargs):
        print(f"DecoderOnlyModelImplBase: load...")
        self.hf_model = hf_model
        self.get_config()
        self.get_embedding()
        self.get_decode_layers()
        self.get_norm()
        self.get_lm_head()
        print(f"DecoderOnlyModelImplBase: load, done.")

    def get_config(self):
        self.hf_config = self.hf_model.config
        print(f"DecoderOnlyModelImplBase: hf_config -> {self.hf_config}")

    def get_embedding(self):
        self.embedding = self.hf_model.model.embed_tokens
        print(f"DecoderOnlyModelImplBase: embedding -> {self.embedding}")

    def get_decode_layers(self):
        self.decode_layers = self.hf_model.model.layers
        print(f"DecoderOnlyModelImplBase: decode_layers -> {self.decode_layers}")

    def get_norm(self):
        self.norm = self.hf_model.model.norm
        print(f"DecoderOnlyModelImplBase: norm -> {self.norm}")

    def get_lm_head(self):
        self.lm_head = self.hf_model.lm_head
        print(f"DecoderOnlyModelImplBase: lm_head -> {self.lm_head}")

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                **kwargs):
        # id to embedding
        if inputs_embeds is None:
            hidden_states = self.embedding(input_ids)
        else:
            print("input is embed.")
            hidden_states = inputs_embeds
        # decode layers and norm
        past_key_values_out = []
        for i, layer in enumerate(self.decode_layers):
            hidden_states, past_key_value_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=False,
                use_cache=True)
            # print(f"past_key_value_out[0] shape: {past_key_value_out[0].shape}")
            past_key_values_out.append(past_key_value_out)
        hidden_states = self.norm(hidden_states)
        # logit
        print(f"hidden_states shape: {hidden_states.shape}")
        logit = self.lm_head(hidden_states).float()
        print(f"logit shape: {logit.shape}")
        print(f"past_key_values_out len: {len(past_key_values_out)}")
        return logit, *past_key_values_out


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


class ModelHfCreatorBase(object):
    def __init__(self):
        super(ModelHfCreatorBase, self).__init__()

    def get_hf_model(self, hf_model_path):
        return AutoModelForCausalLM.from_pretrained(hf_model_path).eval()

    def get_tokenizer(self, tokenizer_path):
        return AutoTokenizer.from_pretrained(tokenizer_path)


class ModelHfCreatorFactory(object):
    def __init__(self):
        super(ModelHfCreatorFactory, self).__init__()
        self._model_hf_creator_map = dict()

    def register_hf_creator(self, type_name):
        def _register(target):
            print(f"register hf creator type: {type_name}, target: {target}")
            self._model_hf_creator_map[type_name] = target
            return target
        return _register

    def get_hf_creator(self, type_name):
        return self._model_hf_creator_map[type_name]

    def hf_creator_list(self):
        return list(self._model_hf_creator_map.keys())


MODEL_HF_CREATOR_FACTORY = ModelHfCreatorFactory()


class ModelExportHelperBase(object):
    def __init__(self, hf_config=None, seq_len=None, kv_cache_max_len=None, is_dynamic_shape=False):
        super(ModelExportHelperBase, self).__init__()
        self.hf_config = hf_config
        self.seq_len = seq_len
        self.kv_cache_max_len = kv_cache_max_len
        self.is_dynamic_shape = is_dynamic_shape
        self.head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
        print(f"ModelExportHelperBase: seq_len -> {self.seq_len}, kv_cache_max_len -> {self.kv_cache_max_len}, "
              f"is_dynamic_shape - {self.is_dynamic_shape}, head_dim -> {self.head_dim}")

    def get_model_example_inputs(self):
        input_examples_name = ["input_ids", "attention_mask", "position_ids"]
        input_examples_tensor = [
            torch.ones(1, self.seq_len, dtype=torch.int32),
            torch.ones(1, 1, self.seq_len, self.kv_cache_max_len, dtype=torch.float32),
            torch.ones(1, self.seq_len, dtype=torch.int32),
        ]
        past_key_value_example_tensors = []
        for i in range(0, self.hf_config.num_hidden_layers):
            input_examples_name.extend([f"past_key{i}", f"past_value{i}"])
            past_key_value_example_tensors.append([
                torch.ones(1, self.hf_config.num_key_value_heads, self.kv_cache_max_len - self.seq_len,
                           self.head_dim, dtype=torch.float32),
                torch.ones(1, self.hf_config.num_key_value_heads, self.kv_cache_max_len - self.seq_len,
                           self.head_dim, dtype=torch.float32),
            ])
        input_examples_tensor.append(past_key_value_example_tensors)
        return input_examples_name, input_examples_tensor

    def get_model_example_outputs(self):
        output_examples_name = ["logit"]
        for i in range(0, self.hf_config.num_hidden_layers):
            output_examples_name.extend([f"past_key_out{i}", f"past_value_out{i}"])
        return output_examples_name


class ModelExportHelperFactory(object):
    def __init__(self):
        super(ModelExportHelperFactory, self).__init__()
        self._model_export_helper_map = dict()

    def register_export_helper(self, type_name):
        def _register(target):
            print(f"register export helper type: {type_name}, target: {target}")
            self._model_export_helper_map[type_name] = target
            return target
        return _register

    def get_export_helper(self, type_name):
        return self._model_export_helper_map[type_name]

    def export_helper_list(self):
        return list(self._model_export_helper_map.keys())


MODEL_EXPORT_HELPER_FACTORY = ModelExportHelperFactory()
