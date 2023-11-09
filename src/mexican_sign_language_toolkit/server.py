import glob

import gradio as gr

from mexican_sign_language_toolkit.pipeline import pipeline, VideoPipeline


def predict(path):
    predict_with_path = pipeline(VideoPipeline())
    return predict_with_path(path)


def launch_server():
    demo = gr.Interface(predict, [gr.Video(sources="webcam", mirror_webcam=False)], ["text"],
                        examples=[*glob.glob('**/*.mp4', recursive=True), *glob.glob('**/*.mkv', recursive=True)],
                        title="Mexican sign language recognizer demo",
                        description="Mexican sign language recognizer by Ricardo Morfin, Ernesto Lozano, and Carlos Sanchez", )
    demo.launch(share=True)
