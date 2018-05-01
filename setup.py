import os
import shutil

if os.path.exists('models'):
  shutil.rmtree(tensorboard_path)
if os.path.exists('checkpoints'):
  shutil.rmtree(tensorboard_path)

os.mkdir('models')
os.mkdir('checkpoints')
