import json
import os
import warnings

import torch

from ..configs.model_config import MODEL_LOADER_CONFIGS
from .lora import GeneralLoRAFromPeft
from .utils import hash_state_dict_keys, init_weights_on_device, load_state_dict, merge_lora_weights


def load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(f"    This model is initialized with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}
        torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()
        model.load_state_dict(model_state_dict, assign=True)
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(self, keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (model_names, model_classes, model_resource)

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False

    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            loaded_model_names, loaded_models = load_model_from_single_file(
                state_dict, model_names, model_classes, model_resource, torch_dtype, device
            )
            return loaded_model_names, loaded_models

        # Load models without strict matching
        # (the shape of parameters may be inconsistent, and the state_dict_converter will modify the model architecture)
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            loaded_model_names, loaded_models = load_model_from_single_file(
                state_dict, model_names, model_classes, model_resource, torch_dtype, device
            )
            return loaded_model_names, loaded_models

        return loaded_model_names, loaded_models


class ModelManager:
    def __init__(self, torch_dtype=torch.float16, device="cuda", file_path_list=None):
        self.torch_dtype = torch_dtype
        self.device = device

        self.model = []
        self.model_path = []
        self.model_name = []

        self.model_detector = ModelDetectorFromSingleFile(MODEL_LOADER_CONFIGS)

        if file_path_list is not None:
            self.load_models(file_path_list)

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"[ModelManager] Loading models from: {file_path}")

        device = device or self.device
        torch_dtype = torch_dtype or self.torch_dtype

        if isinstance(file_path, list):
            state_dict = {}
            for _file_path in file_path:
                state_dict.update(load_state_dict(_file_path))
        else:
            assert os.path.isfile(file_path)
            state_dict = load_state_dict(file_path)

        if self.model_detector.match(file_path, state_dict):
            model_names, models = self.model_detector.load(
                file_path,
                state_dict,
                device=device,
                torch_dtype=torch_dtype,
                allowed_model_names=model_names,
                model_manager=self,
            )
            for model_name, model in zip(model_names, models):
                self.model.append(model)
                self.model_path.append(file_path)
                self.model_name.append(model_name)
            print(f"[ModelManager] The following models are loaded: {model_names}.")
        else:
            warnings.warn(f"[ModelManager] Cannot load the model from: {file_path}")

    def load_dit(self, ckpt_path: str, device=None, torch_dtype=None, lora_alpha=1.0):
        from .wan_video_dit import WanModel

        device = device or self.device
        torch_dtype = torch_dtype or self.torch_dtype

        state_dict = merge_lora_weights(ckpt_path=ckpt_path, lora_alpha=lora_alpha)

        model_config_path = os.path.join(ckpt_path[:ckpt_path.find("/checkpoints/")], "model_config.json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        with init_weights_on_device():
            model = WanModel(**model_config)
        model.load_state_dict(state_dict, assign=True)
        model = model.to(dtype=torch_dtype, device=device)
        model.eval()

        model_name = "wan_video_dit"
        self.model.append(model)
        self.model_path.append(ckpt_path)
        self.model_name.append(model_name)

        print(f"[ModelManager] The following models are loaded: {[model_name]}.")

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names, device=device, torch_dtype=torch_dtype)

    def fetch_model(self, model_name, file_path=None, require_model_path=False):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"[ModelManager] No `{model_name}` models available.")
            return None
        if len(fetched_models) == 1:
            print(f"[ModelManager] Using `{model_name}` from {fetched_model_paths[0]}.")
        else:
            print(
                f"[ModelManager] More than one `{model_name}` models are loaded in model manager: {fetched_model_paths}. "
                + f"Using `{model_name}` from {fetched_model_paths[0]}."
            )
        if require_model_path:
            return fetched_models[0], fetched_model_paths[0]
        else:
            return fetched_models[0]

    def to(self, device):
        for model in self.model:
            model.to(device)
