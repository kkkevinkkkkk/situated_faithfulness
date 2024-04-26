import os
from typing import Union, Optional, Callable

import torch
from transformers import LlamaForSequenceClassification, PretrainedConfig
from peft import LoraConfig, PeftModel, PeftModelForSequenceClassification



class MyLlamaForSequenceClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        return super().forward(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if not os.path.exists(os.path.join(pretrained_model_name_or_path, "clf.bin")):
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
            return model

        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        config_path = config if config is not None else pretrained_model_name_or_path


        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )

        base_model_path = config._name_or_path
        # load the saved layer and load the base model
        model = super().from_pretrained(
            base_model_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        classifier_path = os.path.join(pretrained_model_name_or_path, "clf.bin")
        classifier_state_dict = torch.load(classifier_path)
        model.score.load_state_dict(classifier_state_dict)
        print(f"Loaded classifier from {classifier_path}")

        return model


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        # only save the final layer
        state_dict = self.state_dict()
        clf_state_dict = {k.split(".")[-1]: v for k, v in state_dict.items() if "score" in k}
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(clf_state_dict, os.path.join(save_directory, "clf.bin"))
        print("save classifier to ", os.path.join(save_directory, "clf.bin"))

        state_dict = {}

        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )

