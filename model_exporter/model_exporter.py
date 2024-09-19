import torch
import transformers
import onnx
import onnxsim
from onnxsim import simplify
import onnxruntime
import os
import shutil

from .compress_model import compress_onnx_model, uncompress_onnx_model

from . import model_base
from . import model_tiny_llama


class ModelExporter(object):
    def __init__(self, seq_len=None, kv_cache_max_len=None, is_dynamic_shape=False):
        super(ModelExporter, self).__init__()
        self.tokenizer_model = None
        self.hf_model = None
        self.model_impl = None
        self.max_gen_tokens = 8
        self.seq_len = seq_len
        self.kv_cache_max_len = kv_cache_max_len
        self.is_dynamic_shape = is_dynamic_shape
        self.export_helper = None
        self.model_type = None

    @staticmethod
    def support_model_type_list():
        impl_model_type_list = model_base.MODEL_IMPL_FACTORY.model_type_list()
        hf_creator_type_list = model_base.MODEL_HF_CREATOR_FACTORY.hf_creator_list()
        export_helper_type_list = model_base.MODEL_EXPORT_HELPER_FACTORY.export_helper_list()
        assert impl_model_type_list == hf_creator_type_list, f"impl_model_type_list != hf_creator_type_list\n" \
                                                             f"impl_model_type_list -> {impl_model_type_list}\n" \
                                                             f"hf_creator_type_list -> {hf_creator_type_list}"
        assert impl_model_type_list == export_helper_type_list, f"impl_model_type_list != export_helper_type_list\n" \
                                                                f"impl_model_type_list -> {impl_model_type_list}\n" \
                                                                f"export_helper_type_list -> {export_helper_type_list}"
        return impl_model_type_list

    def load(self, hf_model_path, model_type, **kwargs):
        # 1. load tokenizer
        hf_creator = model_base.MODEL_HF_CREATOR_FACTORY.get_hf_creator(model_type)()
        print(f"exporter: loading tokenizer...")
        self.tokenizer_model = hf_creator.get_tokenizer(hf_model_path)
        print(f"exporter: load tokenizer, done.")
        # 2. only support hf model, other model is to supported
        # load hf model
        print(f"exporter: loading hf model...")
        self.hf_model = hf_creator.get_hf_model(hf_model_path)
        print(f"exporter: load hf model, done.")
        # 3. create impl
        print(f"exporter: creating model impl...")
        self.model_impl = model_base.MODEL_IMPL_FACTORY.get_model(model_type)()
        self.model_impl.load(self.hf_model)
        print(f"exporter: create model, done.")
        # 4. export helper
        print(f"exporter: preparing export helper...")
        self.export_helper = model_base.MODEL_EXPORT_HELPER_FACTORY.get_export_helper(model_type)(
            hf_config=self.hf_model.config, seq_len=self.seq_len, kv_cache_max_len=self.kv_cache_max_len)
        print(f"exporter: prepare export helper, done.")

    def text_to_ids(self, prompt):
        # torch tensor
        return self.tokenizer_model(prompt, return_tensors="pt")["input_ids"]

    def ids_to_text(self, ids):
        return self.tokenizer_model.decode(ids)

    def get_hf_model_output(self, input_ids):
        hf_model_output = self.hf_model.generate(input_ids,
                                                 do_sample=False,
                                                 max_length=input_ids.shape[1] + self.max_gen_tokens)
        hf_model_output_ids = hf_model_output.tolist()[0]
        hf_model_output_text = self.ids_to_text(hf_model_output_ids)
        return hf_model_output_ids, hf_model_output_text

    def get_impl_model_output(self, input_ids):
        output_ids = input_ids.tolist()[0]
        # prefill
        print("get_impl_model_output: prefilling...")
        seq_len = input_ids.shape[1]
        print(f"get_impl_model_output: input_ids -> {input_ids}, seq_len -> {seq_len}")
        attention_mask = torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0)).view(1, 1, seq_len, seq_len)
        print(f"get_impl_model_output: attention_mask -> {attention_mask}")
        position_ids = torch.arange(0, seq_len).view(1, seq_len)
        print(f"get_impl_model_output: position_ids -> {position_ids}")
        past_key_values = None
        model_output = self.model_impl(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       past_key_values=past_key_values,
                                       inputs_embeds=None)
        past_key_values = model_output[1:]
        next_token = torch.argmax(model_output[0][:, -1, :], dim=-1)
        output_ids.append(next_token.tolist()[0])
        print(f"get_impl_model_output: prefill, done. first token is -> {next_token}")
        # decode
        print("get_impl_model_output: decoding...")
        for i in range(seq_len, seq_len + self.max_gen_tokens - 1):
            input_ids = torch.ones(1, 1, dtype=torch.int32)
            input_ids[0][0] = next_token
            print(f"get_impl_model_output: input_ids -> {input_ids}")
            attention_mask = torch.zeros(1, 1, 1, i + 1, dtype=torch.float32)
            print(f"get_impl_model_output: attention_mask -> {attention_mask}")
            position_ids = torch.arange(i, i + 1).view(1, 1)
            print(f"get_impl_model_output: position_ids -> {position_ids}")
            model_output = self.model_impl(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           past_key_values=past_key_values,
                                           inputs_embeds=None)
            past_key_values = model_output[1:]
            next_token = torch.argmax(model_output[0][:, -1, :], dim=-1)
            output_ids.append(next_token.tolist()[0])
            print(f"get_impl_model_output: decode token is -> {next_token}")
        print("get_impl_model_output: decode, done.")
        return output_ids, self.ids_to_text(output_ids)

    def get_onnx_model_output(self, input_ids):
        # TODO:
        pass

    def export_onnx(self, onnx_model_path, need_test=False, prompt="hello"):
        # test before, hf and impl
        if need_test is True:
            input_ids = self.text_to_ids(prompt)
            print(f"exporter: test prompt is -> {prompt}, ids is -> {input_ids}, "
                  f"max gen tokens is -> {self.max_gen_tokens}")
            # hf model golden
            hf_model_output_ids, hf_model_output_text = self.get_hf_model_output(input_ids)
            print(f"exporter: hf model output ids is -> {hf_model_output_ids}, \n"
                  f"hf model output text is -> {hf_model_output_text}")
            # impl model check
            impl_model_output_ids, impl_model_output_text = self.get_impl_model_output(input_ids)
            print(f"exporter: impl model output ids is -> {impl_model_output_ids}, \n"
                  f"impl model output text is -> {impl_model_output_text}")
            assert hf_model_output_ids == impl_model_output_ids, f"hf_model_output_ids != impl_model_output_ids\n" \
                                                                 f"hf_model_output_ids -> {hf_model_output_ids}\n" \
                                                                 f"impl_model_output_ids -> {impl_model_output_ids}"
            print(f"exporter: hf model is same to impl model.")
        # prepare
        shutil.rmtree(onnx_model_path, ignore_errors=True)
        os.mkdir(onnx_model_path, 777)
        onnx_tmp_path = f"{onnx_model_path}/tmp"
        os.mkdir(onnx_tmp_path, 777)
        example_inputs_name, example_inputs_tensor = self.export_helper.get_model_example_inputs()
        example_outputs_name = self.export_helper.get_model_example_outputs()
        # export onnx model
        print(f"exporter: torch onnx export doing...")
        torch.onnx.export(
            self.model_impl,
            tuple(example_inputs_tensor),
            f"{onnx_tmp_path}/model.onnx",
            input_names=example_inputs_name,
            output_names=example_outputs_name,
        )
        print(f"exporter: torch onnx export, done.")
        # simplify
        print(f"exporter: onnx simplify doing...")
        onnx_model = onnx.load(f"{onnx_tmp_path}/model.onnx")
        size_th_kb = 1024
        size_th_bytes = size_th_kb * 1024
        onnx_model, removed_nodes = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
        tensor_size_threshold = f"{size_th_kb}KB"
        skipped_optimizers = []
        onnx_model, check = simplify(onnx_model,
                                     tensor_size_threshold=tensor_size_threshold,
                                     skipped_optimizers=skipped_optimizers)
        assert check, "simplify failed."
        onnx_model = uncompress_onnx_model(onnx_model, removed_nodes)
        onnx.save(
            onnx_model,
            f"{onnx_model_path}/model.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb"
        )
        print(f"exporter: onnx simplify, done.")
        shutil.rmtree(onnx_tmp_path, ignore_errors=True)
        # test after, onnx and impl
        # TODO:


