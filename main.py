import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from smolagents import CodeAgent, load_tool,tool,  Model, FinalAnswerTool, LiteLLMModel
from custom_model import CustomTransformersModel
from typing import Optional
from utils_tools import get_raw_description_from_product_url, get_price_from_product_url, make_a_search_on_amazon, compare_products, ParserProductDescriptionWithoutGuideTool, ParserProductDescriptionWithGuideTool
from Gradio_UI import GradioUI
os.environ['HF_HOME'] = '/home/ayoub/llm_models'
os.environ['GEMINI_API_KEY'] = 'AIzaSyDdMKL9vyRC3xEnjitbQhaRmpHrYJbgAlo'


SYSTEM_PROMPT_GUIDED = """Tu es un assistant expert de l'extraction d'information. Tu t'appuie sur la description produit fournis par l'utilisateur pour retrouver la liste des attributs qui t'es demandé.
Une fois les avoir extrait, tu structure ta réponse sous le format d'un json avec en clef le nom de l'attribut et en valeur sa valeur. Si jamais un des attribut demandé par l'utilisateur ne se retrouve pas dans la description, tu mets la valeur "N/A".
Par exemple, si l'utilisateur te fournis la fiche d'un canapé et te demande de retrouver les dimensions, la couleur, la matiere et si il est convertible, tu donneras une réponse sous la forme :
{"dimension": dimension du produit,
 "couleur": couleur du produit,
 "matiere": matière du produit,
 "convertible": si oui ou non il est convertible}
N'oublie pas que le client a besoin de toi donc donne toi à fond ! """

model = LiteLLMModel(model_id='gemini/gemini-1.5-flash')

product_description_parser_with_guide = ParserProductDescriptionWithGuideTool(model, SYSTEM_PROMPT_GUIDED)
final_answer = FinalAnswerTool()

agent = CodeAgent(
    tools=[
        # get_weather,
        get_raw_description_from_product_url,
        get_price_from_product_url,
        product_description_parser_with_guide,
        make_a_search_on_amazon,
        compare_products,
        final_answer
    ],
    model=model,
    max_steps=3,
    verbosity_level=2,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    additional_authorized_imports=['pandas', 'json']
)

if __name__ == '__main__':
    GradioUI(agent).launch()