#%%
#load /samoa/data/smrserraoseabr/subsurface-DA-with-generative-models/trained_preprocessors/input_preprocessor_32.pkl
import pickle
import numpy as np
import pandas as pd

#load pickle file
with open('/samoa/data/smrserraoseabr/subsurface-DA-with-generative-models/model_metrics.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
    
# %%
preprocessor
# %%
