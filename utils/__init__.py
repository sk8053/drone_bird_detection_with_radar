import torch
from utils.config_singleton import Config

conf = Config()

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
