import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = parameter.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow = {name: value.detach().clone() for name, value in state_dict["shadow"].items()}
