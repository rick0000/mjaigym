import yaml
import pprint
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class YamlConfig:
    def save(self, config_path: Path):
        if not config_path.parent.is_dir():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "wt") as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def load(cls, config_path: Path):
        with open(config_path, "rt") as f:
            conf_dic = yaml.safe_load(f)
            if conf_dic is None:
                conf_dic = {}
            return cls(**conf_dic)


@dataclass(frozen=True)
class GeneratorConfig(YamlConfig):
    game_type:str = "tonpu"
    use_dfs:bool = False
    one_batch_game_num:int = 2
    train_game_num:int = -1
    dahai_ignore:bool = False


@dataclass(frozen=True)
class ModelConfig(YamlConfig):
    resnet_repeat:int = 20
    mid_channels:int = 128
    learning_rate:float = 0.0001
    batch_size:int = 128
    