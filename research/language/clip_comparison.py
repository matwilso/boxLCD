# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from research.utils import to_np
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

MODEL_NAME = "openai/clip-vit-base-patch32"
#MODEL_NAME = "openai/clip-vit-large-patch14-336"
#MODEL_NAME = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
import ipdb; ipdb.set_trace()

# In[2]:
with open('./descriptions.txt', 'r') as f:
    descs = f.read().splitlines()
descs = [d for d in descs if not d.startswith('#')]

img_path = Path('./example_imgs')
imgs = {x.name: Image.open(x) for x in img_path.glob('*.png')}

inputs = processor(text=descs, images=list(imgs.values()), return_tensors="pt", padding=True)
outputs = model(**inputs)

out = to_np(outputs.logits_per_text)
ax = plt.imshow(out)
ax.axes.set_xticks(np.arange(len(imgs)))
ax.axes.set_xticklabels(imgs.keys(), rotation='vertical')

ax.axes.set_yticks(np.arange(len(descs)))
ax.axes.set_yticklabels(descs)
plt.show()
# %%
