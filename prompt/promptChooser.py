import torch
from torch import nn
from prompt.text_components import TextComponentBank
REAL_NAME = {'all': 'medical image', 'Brain': 'Brain', 'Liver':'Liver', 'Retina_RESC':'retinal OCT', 'Chest':'Chest X-ray film', 'Retina_OCT2017':'retinal OCT', 'Histopathology':'histopathological image'}

class DummyOptimizer:
    def step(self):
        pass  # Do nothing

    def zero_grad(self):
        pass  # Do nothing

class PromptChooser(nn.Module):
    def __init__(self,
                 clip_model,
                 args,
                 device
                 ):
        super(PromptChooser, self).__init__()
        self.lr =  args.learning_rate
        self.component_count = int(getattr(args, "component_count", 6))
        self.component_bank = TextComponentBank(
            clip_model=clip_model,
            obj_name=REAL_NAME[args.obj],
            dataset_key=args.obj,
            device=device,
            component_count=self.component_count,
            n_ctx=int(getattr(args, "n_ctx", 8)),
            class_token_position=["end", "front", "middle"],
            text_adapt_until=int(getattr(args, "text_adapt_until", 0)),
            text_adapt_weight=float(getattr(args, "text_adapt_weight", 0)),
            text_proj_trainable=bool(getattr(args, "text_proj_trainable", 0)),
            llm_prompt=bool(getattr(args, "llm_prompt", False)),
            llm_prompt_path=str(getattr(args, "llm_prompt_path", "")),
        ).to(device)
        self.component_bank.train()

        params = self.component_bank.trainable_parameters()
        if len(params) == 0:
            self.text_optimizer = DummyOptimizer()
        else:
            self.text_optimizer = torch.optim.Adam(
                [{"params": params, "lr": self.lr}],
                betas=(0.5, 0.999),
            )


    def forward(self):
        assert self.component_bank is not None
        return self.component_bank.as_text_features()


    def save_prompt(self, save_dict):
        assert self.component_bank is not None
        save_dict['text_component_bank'] = self.component_bank.state_dict()
        save_dict['component_count'] = self.component_count

        return save_dict
