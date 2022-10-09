import numpy as np
import gradio as gr

demo = gr.Blocks()


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")
        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch(debug=True)

def adjust(x):
    if x < 0:
        return 2 * x + 1
    return 2 * x - 1


def sa2(s):
    res = [{"score": 5, "label": 2}]
    return [adjust(-1 * r['score']) if r['label'] == 'negative' else adjust(r['score']) for r in res]


def get_examples():
    # return [e for e in  open("examplesTR.csv").readlines()]
    return ["Bu filmi beğenmedim\n bu filmi beğendim\n ceketin çok güzel\n bugün ne yesek"]



def grfunc(comments):
    df = pd.DataFrame()
    c2 = [s.strip() for s in comments.split("\n") if len(s.split()) > 2]
    df["scores"] = sa2(c2)
    df.plot(kind='hist')
    return plt.gcf()

iface = gr.Interface(
    fn=grfunc,
    inputs=[gr.inputs.Textbox(placeholder="put your sentences line by line", lines=5), gr.File(), gr.Dataframe(headers=['title', 'author', 'text'])],
    outputs="plot")
iface.launch()