import os
import re
from typing import Optional
import pandas as pd
import json

from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available
from src.aiagent.utils.rending_method import cleaning_model_thinking

from src.aiagent.ui.amazon_gradio_theme import AmazonTheme

theme_amazon = AmazonTheme()


def pull_messages_from_step(
        step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield step_number, "STEP_NUMBER"
        # yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()

            yield model_output, "STEP"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:

            yield str(step_log.error), "ERROR"

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """

        yield step_footnote, 'FOOTNOTE'


def stream_to_gradio(
        agent,
        task: str,
        reset_agent_memory: bool = False,
        additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message, status_code in pull_messages_from_step(
                step_log,
        ):
            yield message, status_code

    final_answer = step_log  # Last log is the run's final_answer

    yield final_answer, "FINAL"


def generate_product_cards(l_products):
    cards = ""
    for product in l_products:
        details = ""
        for feature in product.keys():
            if feature not in ["product_name", "image_url", "product_link"]:
                details += f"<li><b>{feature.capitalize()}:</b> {product[feature]}</li>"

        image_link = product.get("image_url", 'assets/no_image.png')
        product_name = product.get("product_name", 'product name')
        product_link = product.get("product_link", 'www.amazon.fr')

        card = f"""
        <div style='flex: 0 0 auto; width: 200px; border: 1px solid #ddd; border-radius: 10px; padding: 10px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1); text-align: center; background: white; transition: transform 0.2s;'>
            <a href="{product_link}" target="_blank" style="text-decoration: none; color: inherit;">
                <img src="{image_link}" style='width: 100%; height: 250px; object-fit: scale-down; border-radius: 8px;' />
                <h4 style='margin: 10px 0 5px;'>{product_name}</h4>
            </a>
            <ul style='list-style: none; padding: 0; font-size: 14px; text-align: left;'>{details}</ul>
        </div>
        """
        cards += card
    return f"<div style='display: flex; overflow-x: auto; gap: 16px; padding: 10px;'>{cards}</div>"


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        self.cmt = cleaning_model_thinking(model_name='gemini/gemini-2.0-flash')
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, memory_product):
        import gradio as gr

        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages, memory_product

        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):

            # if msg[1] == "STEP_NUMBER":
            #     messages.append(gr.ChatMessage(role="assistant", content=f"**{msg[0]}**"))

            if msg[1] == "STEP":
                preprocessed_message = self.cmt(msg[0])
                messages.append(
                    gr.ChatMessage(role="assistant", content=preprocessed_message))

            if msg[1] == "ERROR":
                messages.append(
                    gr.ChatMessage(role="assistant", content=msg[0], metadata={"title": "💥 Error"}))

            if msg[1] == "FOOTNOTE":
                messages.append(
                    gr.ChatMessage(role="assistant", content=msg[0]))
                messages.append(gr.ChatMessage(role="assistant", content="-----"))

            if msg[1] == "FINAL":

                final_answer = msg[0]

                messages.append(
                    gr.ChatMessage(role="assistant", content=str(final_answer[0])))

                if isinstance(final_answer[1], pd.DataFrame):
                    l_product = json.loads(final_answer[1].to_json(orient="records"))
                    html_product = generate_product_cards(l_product)
                    memory_product = html_product

                elif isinstance(final_answer[1], list):
                    html_product = generate_product_cards(final_answer[1])
                    memory_product = html_product

            yield messages, memory_product
        yield messages, memory_product

    def log_user_message(self, text_input):
        return text_input, ""

    def reset_agent(self):
        self.agent.memory.reset()
        self.agent.monitor.reset()

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(theme=theme_amazon, fill_height=True) as demo:
            stored_messages = gr.State([])
            current_product = gr.State([])

            with gr.Row(scale=1):
                gr.Markdown('')

                with gr.Column(scale=1):
                    clear = gr.Button("Reset conversation", variant="primary")

                with gr.Column(scale=10):
                    chatbot = gr.Chatbot(
                        label="🤖 AmazAgent",
                        show_label=False,
                        type="messages",
                        avatar_images=(
                            "assets/user_image.png",
                            "assets/logo_amazon_circle.png",
                        ),
                        resizeable=True,
                        scale=1,
                    )

                    thinking_message = gr.Markdown("🤖 Let me think...", visible=False)
                    product_reco = gr.HTML()
                    with gr.Row():
                        text_input = gr.Textbox(lines=1,
                                                label="Chat Message",
                                                show_label=False,
                                                placeholder="I'm looking for a black dress for a wedding that will take place Saturday next week. Can you help me?",
                                                container=False,
                                                scale=15)
                        search_button = gr.Button("🔍︎", scale=1, variant="secondary")

                    text_input.submit(
                        self.log_user_message,
                        inputs=text_input,
                        outputs=[stored_messages, text_input],
                    ).then(
                        lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=thinking_message,
                    ).then(
                        self.interact_with_agent,
                        inputs=[stored_messages, chatbot, current_product],
                        outputs=[chatbot, product_reco],
                    ).then(
                        lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=thinking_message,
                    )

                    search_button.click(
                        self.log_user_message,
                        inputs=text_input,
                        outputs=[stored_messages, text_input],
                    ).then(
                        lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=thinking_message,
                    ).then(
                        self.interact_with_agent,
                        inputs=[stored_messages, chatbot, current_product],
                        outputs=[chatbot, product_reco],
                    ).then(
                        lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=thinking_message,
                    )

                clear.click(self.reset_agent, None, None
                            ).then(lambda: None, None, chatbot
                                   ).then(lambda: None, None, current_product
                                          ).then(lambda: None, None, product_reco)

        demo.launch(debug=True, share=True, **kwargs)


__all__ = ["stream_to_gradio", "GradioUI"]
