import os
from smolagents import CodeAgent, LiteLLMModel
from src.aiagent.utils.ecom_tools import (search_on_amazon,
                                          ParserProductDescriptionWithGuideTool, GetProductDescriptionTool,
                                          CompareProductTool, FilterProduct, FinalAnswerTool)
from src.aiagent.ui.main_gradio import GradioUI
import yaml
from src.aiagent.core.custom_python_executor import LocalPythonExecutor

with open('config/secrets.yaml') as f:
    SECRETS = yaml.safe_load(f)

os.environ['GEMINI_API_KEY'] = SECRETS['gemini_token']

tools_model = LiteLLMModel(model_id='gemini/gemini-2.0-flash') # Model used for tools using llm
reasoning_model = LiteLLMModel(model_id='gemini/gemini-2.0-flash') # Model used for reasoning

# Tools initialization
product_description_parser_with_guide = ParserProductDescriptionWithGuideTool(tools_model)
compare_products = CompareProductTool(tools_model)
filter_product = FilterProduct(tools_model)
get_product_description = GetProductDescriptionTool()
final_answer = FinalAnswerTool()

# Loading system prompt
template = yaml.safe_load(open("config/prompt.yaml"))

# agent initialization
agent = CodeAgent(
    tools=[
        get_product_description,
        product_description_parser_with_guide,
        search_on_amazon,
        compare_products,
        filter_product,
        final_answer
    ],
    model=reasoning_model,
    prompt_templates=template,
    max_steps=8,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    additional_authorized_imports=['pandas', 'json']
)

# Using customized python executor for accepting double parameters in the final answer
agent.python_executor = LocalPythonExecutor(agent.additional_authorized_imports,
                                            max_print_outputs_length=agent.max_print_outputs_length)

if __name__ == '__main__':
    GradioUI(agent).launch(allowed_paths=["assets"])
