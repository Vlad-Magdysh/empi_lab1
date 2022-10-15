import math

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

VALUES_KEY = 'variation'
FREQ_OF_VALUES_KEY = 'frequency'
FREQ_DIVIDE_LEN_KEY = 'relative'
ECDF_KEY = 'ECDF'

def ecdf_calculate(items_list):
    ecdf = []
    val = 0
    for p in items_list:
        val += p
        ecdf.append(val)
    return ecdf

def ecdf_step_plot(x, y):
    fig = plt.figure()
    plt.step(x, y, where='post')
    plt.title('Графік емпіричної функції розподілу')
    plt.ylabel('ECDF')
    plt.xlabel('Values')
    return fig

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

        with gradio.TabItem("Розбиття на класи") as tab:
            classes_input = gradio.Number()
            classes_button = gradio.Button("Change")
            with gradio.Row():
                with gradio.Column():
                    classes_data_frame = gradio.Dataframe()
                with gradio.Column():
                    classes_plot = gradio.Plot()


    def main(file_obj):
        global data_frame, file_data, original_data

        if file_obj is None:
            return {file_output: "Завантажте спочатку файл!"}

        with open(file_obj.name, "r") as source:
            original_data = [float(i) for i in source.read().split(sep="\n")]
            file_data = Counter(original_data)
        logger.info(f"File {file_obj.name} is uploaded")

        data_frame = pd.DataFrame(sorted(file_data.items()), columns=[VALUES_KEY, FREQ_OF_VALUES_KEY])

        data_frame[FREQ_DIVIDE_LEN_KEY] = data_frame[FREQ_OF_VALUES_KEY] / len(original_data)
        data_frame[ECDF_KEY] = ecdf_calculate(data_frame[FREQ_DIVIDE_LEN_KEY])
        fig = ecdf_step_plot(data_frame[VALUES_KEY], data_frame[ECDF_KEY])
        #Initializetion
        pre_classes_calc = classes_calculator()
        return {
            file_output: "Файл успішно завантажено",
            variation_data_frame: data_frame,
            variation_plot: fig,
            classes_data_frame: pre_classes_calc[classes_data_frame],
            classes_plot: pre_classes_calc[classes_plot]
        }

    def classes_calculator(M_=None):
        """
        Divides the data into different_classes
        Step 1: calculate number_of_classes(M), length of a classes(h),
        Step 2: calculate classes borders
        Step 3: classify items on classes
        Step 4: draw ecdf plot
        :param num_of_classes: Number of the classes, may be given by user or calculated as default
        :return: data_frame
        """
        global data_frame, file_data, original_data
        if M_ is None:
            N = len(original_data)
            M_ = 1 + 3.32*math.log(N, 10)
        M_ = int(M_)
        x_min, x_max = min(data_frame[VALUES_KEY]), max(data_frame[VALUES_KEY])
        h = (x_max - x_min) / M_
        all_borders = [x_min + h*i for i in range(M_+1)]
        # class border [a, b)  include a
        classes = {i: (all_borders[i], all_borders[i+1]) for i in range(M_)}
        classes_df = pd.DataFrame(sorted(classes.items()), columns=[VALUES_KEY, 'border'])
        classes_df[FREQ_OF_VALUES_KEY] = {c_num: len([1 for val in data_frame[VALUES_KEY] if c_range[0] <= val < c_range[1] or val == x_max]) for c_num, c_range in classes.items() }
        classes_df[FREQ_DIVIDE_LEN_KEY] = classes_df[FREQ_OF_VALUES_KEY] / len(original_data)
        fig = plt.figure()
        plt.bar(all_borders[:-1], classes_df[FREQ_DIVIDE_LEN_KEY], width=1.0)
        plt.xlim([x_min,x_max])
        plt.title('Гістограма класів')
        plt.ylabel('p')
        plt.xlabel('Values')
        return {
            classes_data_frame: classes_df,
            classes_plot: fig
        }

    file_upload_button.click(
        fn=main,
        inputs=[file_upload],
        outputs=[file_output, variation_data_frame, variation_plot, classes_data_frame, classes_plot]
    )
    classes_button.click(
        fn=classes_calculator,
        inputs=[classes_input],
        outputs=[classes_data_frame, classes_plot]
    )

demo.launch(debug=True)
