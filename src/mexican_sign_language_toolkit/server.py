import gradio as gr
import numpy as np
from mexican_sign_language_toolkit.pipeline import pipeline,VideoPipeline
import glob 


def launch_server():
  predict = pipeline(VideoPipeline())
  demo = gr.Interface(
    predict,
    [gr.Video()], 
    ["text"],
    examples=glob.glob('datasets/videos/*.mp4'),
    title="Mexican sign language demo",
    description="mexican-sign-language-toolkit",
  )
  demo.launch(share=True)
