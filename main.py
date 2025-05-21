import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from smolagents import CodeAgent, load_tool,tool,  Model, LiteLLMModel #UserInputTool #,FinalAnswerTool
from custom_model import CustomTransformersModel
from typing import Optional
from utils_tools import (get_price_from_product_url, search_on_amazon,
                         ParserProductDescriptionWithGuideTool, GetProductDescriptionTool,
                         CompareProductTool, FilterProduct, FinalAnswerTool)
from test_gradio import GradioUI
import yaml
from custom_python_executor import LocalPythonExecutor

os.environ['HF_HOME'] = '/home/ayoub/llm_models'
os.environ['GEMINI_API_KEY'] = 'AIzaSyDdMKL9vyRC3xEnjitbQhaRmpHrYJbgAlo'


SYSTEM_PROMPT_GUIDED = """You are an expert assistant in product information extraction.

Based on a *product description* provided by the user, your job is to identify and extract the *requested attributes*, 
and organize them in a structured JSON format.

Your response must **always include at minimum** the following keys, even if they are not explicitly requested:
- "product_name"
- "image_url"
- "price"

For each requested attribute:
- If it is found in the description, provide its value.
- If it is missing, return `"N/A"` as the value.

Your final output must be a valid JSON object, using the exact attribute names as keys.

Example:
If the description is about a sofa and the requested attributes are ["dimension", "color", "material", "convertible"],
your output should look like this:

{
 "product_name": "Oslo 3-seater Sofa",
 "image_url": "https://...",
 "price": "€499",
 "dimension": "200x90x85 cm",
 "color": "Light grey",
 "material": "Fabric and wood",
 "convertible": "Yes"
}

Be precise, structured, and always do your best to help the customer understand the product clearly."""

model = LiteLLMModel(model_id='gemini/gemini-2.0-flash')

product_description_parser_with_guide = ParserProductDescriptionWithGuideTool(model, SYSTEM_PROMPT_GUIDED)
compare_products = CompareProductTool(model)
filter_product = FilterProduct(model)
get_product_description = GetProductDescriptionTool()
# input_user = UserInputTool()
final_answer = FinalAnswerTool()

template = yaml.safe_load(open("prompt.yaml"))

agent = CodeAgent(
    tools=[
        get_product_description,
        # get_price_from_product_url,
        product_description_parser_with_guide,
        search_on_amazon,
        compare_products,
        filter_product,
        final_answer
    ],
    model=model,
    prompt_templates=template,
    max_steps=8,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    additional_authorized_imports=['pandas', 'json']
)

agent.python_executor = LocalPythonExecutor(agent.additional_authorized_imports,
                                            max_print_outputs_length=agent.max_print_outputs_length)

if __name__ == '__main__':
    GradioUI(agent).launch()