import pandas as pd
import gradio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_data = None
original_data = None
data_frame = None

demo = gradio.Blocks()
with demo:
    gradio.Markdown("ЕМПІ: Лабораторна робота №1")
    with gradio.Box():
        file_upload = gradio.File(interactive=True)
        file_output = gradio.Textbox()
    file_upload_button = gradio.Button("Upload")

    with gradio.Tabs():
        with gradio.TabItem("Варіаційний ряд"):
            gradio.Markdown("Пункти 1 та 2")
            with gradio.Row():
                with gradio.Column():
                    variation_data_frame = gradio.Dataframe()
                with gradio.Column():
                    variation_plot = gradio.Plot()

        with gradio.TabItem("Розбиття") as tab:
            with gradio.Row():
                image_input = gradio.Image()
                image_output = gradio.Image()
            image_button = gradio.Button("Flip")

    def main(file_obj):
        global data_frame, file_data, original_data
        if file_obj is None:
            return {file_output: "Завантажте спочатку файл!"}
        with open(file_obj.name, "r") as source:
            original_data = [float(i) for i in source.read().split(sep="\n")]
            file_data = Counter(original_data)
        logger.info(f"File {file_obj.name} is uploaded")
        data_frame = pd.DataFrame(sorted(file_data.items()), columns=['variation', 'frequency'])

        data_frame['relative'] = data_frame['frequency'] / len(original_data)
        ecdf = []
        val = 0
        for p in data_frame['relative']:
            val += p
            ecdf.append(val)
        data_frame['ECDF'] = ecdf
        fig = plt.figure()
        plt.step(sorted(data_frame['variation']), data_frame['ECDF'])
        plt.title('Графік емпіричної функції розподілу')
        plt.ylabel('ECDF')
        plt.xlabel('Values')

        return {
            file_output: "Файл успішно завантажено",
            variation_data_frame: data_frame,
            variation_plot: fig
        }

    file_upload_button.click(
        fn=main,
        inputs=file_upload,
        outputs=[file_output, variation_data_frame, variation_plot]
    )

demo.launch(debug=True)
