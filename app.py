import gradio as gr


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            "https://m.media-amazon.com/images/I/71T3tmMx2FL.__AC_SY300_SX300_QL70_FMwebp_.jpg",
            "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
        ),
        resizeable=True,
        scale=1,
    )

demo.launch()