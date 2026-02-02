import torch
import torch.nn as nn

import deeplake
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from transformers import BertTokenizer, BertModel
import nltk
import random

from torch.utils.data import Dataset
from PIL import Image

ds = deeplake.load('hub://activeloop/flickr30k')

images = ds.image
captions = ds.caption_0 ## NextSteps: currently only training on caption0, could include other set of captions

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

## ALL Hyperparameters
caption_max_length = 20
vocab_size = len(tokenizer)