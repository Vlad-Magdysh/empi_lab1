import math

import pandas as pd
import gradio
import numpy as np
import matplotlib
import scipy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from collections import Counter, defaultdict
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_data = None
original_data = None
data_frame = None
ANOMALY_REMOVED_TIMES = 0

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

#UI!!!
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
                    probability_paper = gradio.Plot()

        with gradio.TabItem("Розбиття на класи") as tab:
            gradio.Markdown("Пункти 3, 4 ,5")
            classes_input = gradio.Number(label="Classes")
            bandwidth_input = gradio.Number(label="Bandwidth")
            classes_button = gradio.Button("Change")
            with gradio.Row():
                with gradio.Column():
                    classes_data_frame = gradio.Dataframe()
                with gradio.Column():
                    classes_plot = gradio.Plot()
        with gradio.TabItem("Статичні характеристики"):
            gradio.Markdown("Пункт 6")
            with gradio.Row():
                characteristics_output = gradio.Dataframe()
        with gradio.TabItem("Аномалії"):
            gradio.Markdown("Пункт 7")
            anomaly_counter = gradio.Textbox()
            with gradio.Row():
                with gradio.Column():
                    anomalies_list = gradio.Textbox(label="Аномальні значення", lines=6)
                    anomaly_button = gradio.Button("Прибрати аномальні значення")
                with gradio.Column():
                    anomaly_plot = gradio.Plot()



    def main(file_obj, remove_anomaly=False):
        #Варіант 12 Нормальний  ЗГОДА: Колмогорова
        global data_frame, file_data, original_data
        if remove_anomaly:
            anomaly_values = find_anomaly_values()[anomalies_list]
            original_data = [x for x in original_data if x not in anomaly_values]
        else:
            if file_obj is None:
                return {file_output: "Завантажте спочатку файл!"}

            with open(file_obj.name, "r") as source:
                original_data = [float(i) for i in source.read().split(sep="\n")]
                logger.info(f"File {file_obj.name} is uploaded")
        file_data = Counter(original_data)

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
            classes_plot: pre_classes_calc[classes_plot],
            **calculate_characteristics(),
            **find_anomaly_values(),
            **calculate_paper(data_frame[VALUES_KEY], data_frame[FREQ_DIVIDE_LEN_KEY])
        }

    def classes_calculator(M_=None, B_=None):
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
        if not M_:
            N = len(original_data)
            M_ = 1 + 3.32*math.log(N, 10)
        M_ = int(M_)
        x_min, x_max = min(data_frame[VALUES_KEY]), max(data_frame[VALUES_KEY])
        h = (x_max - x_min) / M_
        all_borders = [x_min + h*i for i in range(M_+1)]
        # class border [a, b)  include a
        classes = {i: (all_borders[i], all_borders[i+1]) for i in range(M_)}
        classes_df = pd.DataFrame(sorted(classes.items()), columns=[VALUES_KEY, 'border'])
        a = {c_num: len([1 for val in original_data if c_range[0] <= val < c_range[1] or val == x_max == c_range[1]]) for c_num, c_range in classes.items() }
        classes_df[FREQ_OF_VALUES_KEY] = a
        classes_df[FREQ_DIVIDE_LEN_KEY] = classes_df[FREQ_OF_VALUES_KEY] / len(original_data)

        fig, ax1 = plt.subplots()
        ax1.hist(original_data, bins=all_borders, ec="k")
        y_vals = ax1.get_yticks()
        ax1.set_yticklabels(['{}'.format(round(x/len(original_data), 2)) for x in y_vals])
        plt.xlim([x_min, x_max])
        plt.title('Гістограма класів')
        plt.ylabel('p')
        plt.xlabel('Values')

        #Using Gaussian Kernel
        # density = gaussian_kde(original_data, bw_method='scott')
        #Using Gaussian Kernel
        #Specify bandwidth parameter, using Skott rule
        if not B_:
            density = gaussian_kde(original_data, bw_method='scott')
        else:
            density = gaussian_kde(original_data)
            density.covariance_factor = lambda: B_
            density._compute_covariance()
        plt.plot(sorted(original_data), [i*len(original_data) for i in density(sorted(original_data))])
        #density = gaussian_kde(original_data)

        return {
            classes_data_frame: classes_df,
            classes_plot: fig
        }

    def calculate_characteristics():
        """Calculates characteristics. Returns table as a dataframe"""
        #TODO Refactor and group all parameters
        sorted_original_data = sorted(original_data)
        DATA_LEN = len(sorted_original_data)

        MEAN = sum(sorted_original_data) / DATA_LEN
        if DATA_LEN % 2:
            MEDIAN = sorted_original_data[DATA_LEN//2]
        else:
            MEDIAN = (sorted_original_data[(DATA_LEN//2)-1] + sorted_original_data[DATA_LEN//2])/2
        #Variance block
        D_dyspersiya = S_2_variance = sum([(x - MEAN)**2 for x in sorted_original_data]) / (DATA_LEN-1)
        #Standard deviation
        S_standard_deviation = math.sqrt(S_2_variance)
        #Coefficient of skewness
        COEF_SKEWNESS = scipy.stats.skew(sorted_original_data, bias=True)
        #Coefficient of kurtosis
        COEF_KURTISIS = scipy.stats.kurtosis(sorted_original_data, bias=True)
        #Coefficient of antikurtosis
        COEF_ANTIKURT = 1 / math.sqrt(COEF_KURTISIS + 3)
        MIN, MAX = sorted_original_data[0], sorted_original_data[-1]

        #intervals
        # Інтервал середнього арифметичного mean
        SIGMA_MEAN = S_standard_deviation / math.sqrt(DATA_LEN)
        # В презентації вказано, якщо @=0.05 , t = u = 1.96 де u квартиль стандартного нормального розподілу
        U1a2 = T_STUDENT = 1.96

        mean_min = MEAN - T_STUDENT*SIGMA_MEAN
        mean_max = MEAN + T_STUDENT*SIGMA_MEAN

        med_min = sorted_original_data[round(DATA_LEN//2 - U1a2*math.sqrt(DATA_LEN)/2)]
        med_max = sorted_original_data[round(DATA_LEN//2 + 1 + U1a2*math.sqrt(DATA_LEN)/2)]

        SQRT_SQRT_SIGMA = S_standard_deviation / math.sqrt(2*DATA_LEN)
        dev_min = S_standard_deviation - T_STUDENT*SQRT_SQRT_SIGMA
        dev_max = S_standard_deviation + T_STUDENT*SQRT_SQRT_SIGMA

        # Далі ідуть general формула, але з різними sqrt_of_sqrt_of_sqrt_of_sqrt_of_sqrt_of_sqrt_sigma
        # Формули взяті з презентації 4, сторінка 10, 29.10 в 22:39
        SQRT_SKEW_SIGMA = math.sqrt( (6*(DATA_LEN-2)) / ((DATA_LEN+1)*(DATA_LEN+3)) )
        coef_skew_min = COEF_SKEWNESS - T_STUDENT*SQRT_SKEW_SIGMA
        coef_skew_max = COEF_SKEWNESS + T_STUDENT*SQRT_SKEW_SIGMA

        SQRT_KURTOSIS_SIGMA = math.sqrt( (24*DATA_LEN*(DATA_LEN-2)*(DATA_LEN-3)) / ((DATA_LEN + 1)**2 * (DATA_LEN+3)*(DATA_LEN+5)) )
        coef_kurt_min = COEF_KURTISIS - T_STUDENT * SQRT_KURTOSIS_SIGMA
        coef_kurt_max = COEF_KURTISIS + T_STUDENT * SQRT_KURTOSIS_SIGMA

        # Немає формули в презентації для контрексцесу:(       SQRT_ANTI_KURT_SIGMA =

        char_df = pd.DataFrame(columns=["Характеристики", VALUES_KEY, "sqrt(Square deviation)", "INTERVAL"])
        char_df["Характеристики"] = ["Середнє значення", "Медіана", "Середньоквадратичне відхилення", "Коеф.асиметрії", "Коеф.ексцесу", "Коеф.контрексцесу", "Мінімум", "Максимум"]
        char_df[VALUES_KEY] = [MEAN, MEDIAN, S_standard_deviation, COEF_SKEWNESS, COEF_KURTISIS, COEF_ANTIKURT, MIN, MAX]
        char_df["sqrt(Square deviation)"] = [SIGMA_MEAN, None, SQRT_SQRT_SIGMA, SQRT_SKEW_SIGMA, SQRT_KURTOSIS_SIGMA, None, None, None]
        char_df["INTERVAL"] = [(mean_min, mean_max), (med_min, med_max), (dev_min, dev_max), (coef_skew_min, coef_skew_max), (coef_kurt_min, coef_kurt_max), None, None, None]
        return {characteristics_output: char_df}

    #Пункт 7
    def find_anomaly_values():
        """
        1. Calculate [a; b]
        2. Evaluate data, write anomalies to the list
        3. Draw the plot
        :return: lists and plot
        """
        sorted_original_data = sorted(original_data)
        DATA_LEN = len(sorted_original_data)
        MEAN = sum(sorted_original_data) / DATA_LEN
        S_standart_deviation = math.sqrt(sum([(x - MEAN)**2 for x in sorted_original_data]) / (DATA_LEN-1))
        #U is 1.96, because alpha is 5%
        U1a2 = 1.96
        a = MEAN - U1a2 * S_standart_deviation
        b = MEAN + U1a2 * S_standart_deviation
        RANGE = (a, b)

        anomaly_values = []
        for x in original_data:
            if not a <= x <= b:
                anomaly_values.append(x)

        fig = plt.figure()
        plt.scatter(list(range(DATA_LEN)), original_data)
        plt.title('Графік емпіричної функції розподілу')
        plt.ylabel('Значення x')
        plt.xlabel('Індекси')
        plt.axhline(y=a, color='r', linestyle='-')
        plt.axhline(y=b, color='r', linestyle='-')
        return {
            anomalies_list: anomaly_values,
            anomaly_plot: fig,
            anomaly_counter: f"Size = {DATA_LEN} Anomaly = {ANOMALY_REMOVED_TIMES}"
        }

    def anomaly_main_wrapper(file_obj):
        global ANOMALY_REMOVED_TIMES
        ANOMALY_REMOVED_TIMES += 1
        return main(file_obj, remove_anomaly=True)

    #Пункт 8
    def calculate_paper(X_array, F_x):
        fig = plt.figure()
        # t = list(X_array)
        # z = [scipy.stats.norm.ppf(f_x) for f_x in list(F_x)]
        scipy.stats.probplot(original_data, plot=plt)
        #plt.scatter(t, z)
        return {probability_paper: fig}

    file_upload_button.click(
        fn=main,
        inputs=[file_upload],
        outputs=[file_output, variation_data_frame, variation_plot, classes_data_frame, classes_plot,
                 characteristics_output, anomalies_list, anomaly_plot, anomaly_counter, probability_paper]
    )
    classes_button.click(
        fn=classes_calculator,
        inputs=[classes_input, bandwidth_input],
        outputs=[classes_data_frame, classes_plot]
    )
    anomaly_button.click(
        fn=anomaly_main_wrapper,
        inputs=[file_upload],
        outputs=[file_output, variation_data_frame, variation_plot, classes_data_frame, classes_plot,
                 characteristics_output, anomalies_list, anomaly_plot, anomaly_counter, probability_paper]
    )

demo.launch(debug=True)

#Математичнне сподівання == Середнє арифметичне
