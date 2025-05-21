from smolagents import tool, Tool
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
from typing import Any, Dict, Optional

class ParserProductDescriptionWithGuideTool(Tool):
    name = "parse_product_description_with_guide"
    description = (
        "Use this tool when you are given a *detailed product description* and asked to extract specific *product attributes*. "
        "The tool takes two inputs: the raw product description and a list of target attributes "
        "(e.g., 'dimensions', 'material', 'color', 'convertible', etc.). "
        "It returns a structured JSON object containing the requested information, and always includes: "
        "'product_name', 'image_url', and 'price', even if they are not explicitly requested. "
        "If an attribute is not found in the description, it is marked as 'N/A'. "
        "This tool is ideal for structuring product data from unstructured text."
    )
    inputs = {"product_description": {"type": "string",
                                      "description": "The product description containing every information on the product"},
              "product_feature": {"type": "array",
                                  "description": "The list of feature that should be retrieve from product description"},
              }
    output_type = "string"

    def __init__(self, model,
                 system_prompt,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.system_prompt = system_prompt

    def _preprocessing_message(self, product_description, feature_list):


        messages = [{"role": "system",
                     "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user",
                     "content": [{"type": "text", "text": product_description},
                                 {"type": "text", "text": f"retrieve the following features : {feature_list}"}]}
                    ]
        return messages

    def forward(self, product_description: str, product_feature: list[str]):
        messages = self._preprocessing_message(product_description, product_feature)
        model_output = self.model(messages, response_format={"type": "json_object"}).content

        return json.loads(model_output)


class GetProductDescriptionTool(Tool):
    name = "get_product_description"
    description = ("tool that retrieve the product description and the price of a product from Amazon.com, "
                   "all information is condensed into a raw character string")
    inputs = {"product_url": {"type": "string",
                              "description": "The url link of the product on Amazon.com"}}

    output_type = "string"

    def __init__(self):
        super().__init__()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

        self.proxies = {
            "https": "scraperapi.country_code=us:5628df6812853d345249537321235862@proxy-server.scraperapi.com:8001"
        }

    @staticmethod
    def _clean_product_url(product_url: str) -> str:
        pattern = r"(https://www\.amazon\.[a-z.]+/[^/]+/dp/[^/]+)"
        match = re.search(pattern, product_url)

        return match.group(0)


    def forward(self, product_url: str) -> str:

        product_url = self._clean_product_url(product_url)
        try:
            response = requests.get(product_url, headers=self.headers) # , proxies=self.proxies, verify=False)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            preprocessed_html = ""

            # Extraction de la description
            product_title = soup.find("div", id="titleSection")
            product_img = soup.find('div', id='imgTagWrapperId')
            product_description = soup.find("div", id="productDescription")
            alt_description = soup.find("div", id="feature-bullets")
            seller_description = soup.find("div", id="aplus")
            price = soup.find('span', class_='a-offscreen')

            if product_title:
                preprocessed_html += "-- product title -- \n"
                preprocessed_html += product_title.get_text().strip() + '\n\n'

            if product_img:
                preprocessed_html += "-- image url -- \n"
                preprocessed_html += product_img.find('img')['src'] + '\n\n'

            if product_description:
                preprocessed_html += "-- product description by Website -- \n"
                preprocessed_html += product_description.get_text(strip=True) + '\n\n'

            if alt_description:
                preprocessed_html += "-- additional description -- \n"
                preprocessed_html += alt_description.get_text(strip=True) + '\n\n'

            if seller_description:
                preprocessed_html += "-- product description by Seller -- \n"
                preprocessed_html += seller_description.get_text(strip=True) + '\n\n'

            if price:
                preprocessed_html += "-- price of the product --\n"
                preprocessed_html += price.get_text()

            return preprocessed_html

        except requests.exceptions.RequestException as e:
            return f"Error : {str(e)}"


@tool
def get_price_from_product_url(product_url: str) -> str:
    """
    Function that return the price of the product on Amazon

    Args:
        product_url: The url of the product sold on Amazon.com

    Returns:
        A string containing the price and the currency of the product (e.g. : '134,45$')
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    proxies = {
        "https": "scraperapi.country_code=us:5628df6812853d345249537321235862@proxy-server.scraperapi.com:8001"
    }

    try:
        response = requests.get(product_url, headers=headers, proxies=proxies, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        price = soup.find('span', class_='a-offscreen')

        if price:
            return price.get_text()

        else :
            return 'price not found'

    except requests.exceptions.RequestException as e:
        return f"Error : {str(e)}"


@tool
def search_on_amazon(keyword: str) -> list[dict]:
    """
    function to retrieve a list of products resulting from a search on the amazon search engine. For all these products, we also retrieve the image url, the product price and the hypothetical delivery date.

    Args:
        keyword: the keyword to search for in the search engine

    Returns:
        a list containing one json per product with the following three keys :
            - product_name : title of the product
            - image_url : url of the product's image
            - product_link : url of the product page
            - product_price : the price of the product
            - delivery_date : information on delivery date
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    proxies = {
            "https": "scraperapi.country_code=us:5628df6812853d345249537321235862@proxy-server.scraperapi.com:8001"
        }

    url = f"https://www.amazon.fr/s?k={keyword.replace(' ', '+')}"

    # Envoi de la requête HTTP à Amazon
    response = requests.get(url, headers=headers) #, proxies=proxies, verify=False)

    if response.status_code != 200:
        print("Erreur lors de la récupération de la page")

    # Utilisation de BeautifulSoup pour analyser le HTML de la page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Récupérer les 10 premiers produits
    products = []
    product_elements = soup.find_all('div', role='listitem')

    for product in product_elements[:10]:  # Limiter à 10 produits
        product_json = dict()

        tag_sponsorised = product.find('span', class_='puis-label-popover-default')

        if tag_sponsorised:
            continue

        title_element = product.find('h2')
        image_element = product.find('img', class_='s-image')
        link_element = product.find('a', class_='a-link-normal')
        price_element = product.find("span", class_='a-offscreen')
        delivery_element = product.find('div', {"data-cy" : "delivery-recipe"})

        if title_element :
            product_json['product_name'] = title_element.get_text()

        if image_element :
            product_json['image_url'] = image_element.get('src')

        if link_element:
            product_json['product_link'] = 'https://www.amazon.fr' + link_element['href']

        if price_element:
            product_json['price'] = price_element.get_text().replace("\\xa0", " ").replace("\xa0", " ")

        if delivery_element:
            product_json['delivery_date'] = delivery_element.get_text()

        products.append(product_json)

    return products

# @tool
# def compare_products(features: list[str], products: list[any]) -> pd.DataFrame:
#     """
#     Generate a comparison table (as a pandas DataFrame) from a list of structured product dictionaries.
#
#     This function is used when product data is already structured (e.g., extracted via another tool)
#     and the goal is to present selected features in a clear tabular format for comparison.
#
#     Args:
#         features: List of features to compare (e.g., ['price', 'Screen', 'Processor'])
#         products: List of products as dictionaries, there must necessarily have the key product_name and price (e.g., [{'product_name': 'Product 1', 'price': 500, 'Screen': '15"', 'Processor': 'Intel i5'}, {'product_name': 'Product 2', 'price': 600, 'Screen': '17"', 'Processor': 'Intel i7'}])
#
#     Returns:
#         A pandas DataFrame representing the comparison of the products
#     """
#
#     # Créer une liste pour les lignes du tableau
#     comparison_data = []
#
#     # Ajouter les produits dans les colonnes du tableau
#     for product in products:
#         # Extraire uniquement les caractéristiques pertinentes pour ce produit
#         product_comparison = {feature: product.get(feature, 'N/A') for feature in features}
#         comparison_data.append(product_comparison)
#
#     # Créer un DataFrame pandas à partir de la comparaison
#     df_comparison = pd.DataFrame(comparison_data, columns=features)
#
#     return df_comparison

# @tool
# def compare_products_bis(products: list[any]) -> pd.DataFrame:
#     """
#     Compare products based on specified features and return a dataframe to easily read the comparisons
#
#     Args:
#         products: List of products as dictionaries (e.g., [{'Name': 'Product 1', 'Price': 500, 'Screen': '15"', 'Processor': 'Intel i5'}, {'Name': 'Product 2', 'Price': 600, 'Screen': '17"', 'Processor': 'Intel i7'}])
#
#     Returns:
#         A pandas DataFrame representing the comparison of the products
#     """
#
#     df_comparison = pd.DataFrame(products)
#
#     return df_comparison

# @tool
# def display_product_image(product_url: str) -> Image:
#     """
#     Download the product image from product_url and return a PIL.Image object to display the product
#
#     Args:
#         product_url: the url of the image to be downloaded
#     Returns:
#         A image in format PIL.Image downloaded from url
#     """
#     response = requests.get(product_url)
#
#     if response.status_code != 200:
#         print("Error on downloading image process")
#
#     image = Image.open(BytesIO(response.content))
#     return image
#
#
#
# class UserInputTool(Tool):
#     name = "user_input"
#     description = "Asks for user's input on a specific question"
#     inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
#     output_type = "string"
#
#     def forward(self, question):
#         gr.ChatMessage(role="user",
#                              content='info de fou là incroyable')
#         gr.ChatMessage(role="user",
#                              content=question)
#         # user_input = input(f"{question} => Type your answer here:")
#         return 'pas de reponses utilisateur'

class CompareProductTool(Tool):
    name = "compare_product"
    description = (
        "Generate a comparison table (as a pandas DataFrame) from a list of structured product dictionaries."
        "This function is used when product data is already structured (e.g., extracted via another tool)"
        "and the goal is to present selected features in a clear tabular format for comparison."
    )

    inputs = {"list_product_element": {"type": "array",
                                       "description": """List of products as dictionaries, there must necessarily have the key product_name and price (e.g., [{'product_name': 'Product 1', 'price': 500, 'Screen': '15"', 'Processor': 'Intel i5'}, {'product_name': 'Product 2', 'price': 600, 'Screen': '17"', 'Processor': 'Intel i7'}])"""}
              }
    output_type = "any"

    def __init__(self, model,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def _clean_product_info(self, product_description: dict):
        messages = [{"role": "system",
                     "content": ("Tu es un super assistant très fort pour resumer et structurer des json."
                                 "Je vais te fournir un json et tu vas faire en sorte qu'aucune valeur de clef soit superieur à 50 characère."
                                 "Tu as le droit de resumer pour conserver que les informations principales mais tu nas pas le droit de toucher les champs qui sont des urls, tu les rends tel quel sans les affecter."
                                 "Ne change pas la structure du json, il ne doit pas manquer de clef qui étaient presentent initialement ni etre renommé"
                                 "concernant les champs de livraison, je veux que tu ne conserve que la date la plus courte de livraison sous le format jour mois")},
                    {"role": "user", "content": f"met moi en forme ce json stp : {product_description}"}]
        model_output = self.model(messages, response_format={"type": "json"}).content

        model_output = model_output.strip('```').replace('json', '', 1)
        return json.loads(model_output)

    def forward(self, list_product_element: list[dict]):
        for product_index in range(len(list_product_element)):
            list_product_element[product_index] = self._clean_product_info(list_product_element[product_index])

        return pd.DataFrame(list_product_element)


class FilterProduct(Tool):
    name = "filter_product"
    description = (
        "Filtre une liste de produits selon une condition exprimée par l'utilisateur."
        "La condition est une phrase en langage naturel (ex: 'doit être livré avant le 10 mai', 'prix inférieur à 300€', etc.)."
    )

    inputs = {
        "list_product_element": {
            "type": "array",
            "description": "Liste de produits sous forme de dictionnaires. Ex: [{'product_name': 'A', 'price': 100, 'delivery_date': '5 mai'}]"
        },
        "condition": {
            "type": "string",
            "description": "Condition en langage naturel à respecter (ex: 'doit être livré avant le 10 mai')."
        }
    }

    output_type = "array"

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def _check_condition_with_llm(self, product: dict, condition: str) -> bool:
        messages = [
            {"role": "system", "content": (
                "Tu es un assistant chargé d'évaluer si un produit respecte une condition utilisateur."
                "Tu vas recevoir un produit sous forme de dictionnaire, et une condition."
                "Tu dois répondre uniquement par 'oui' ou 'non' (sans autre explication), selon que le produit satisfait la condition ou non en t'aidant des differents champs du dictionnaire."
                "La réponse doit être exactement 'oui' ou 'non', en minuscules."
            )},
            {"role": "user", "content": f"Produit : {product}\n Condition : {condition}"}
        ]

        result = self.model(messages).content.strip().lower()
        return result == "oui"

    def forward(self, list_product_element: list[dict], condition: str):
        filtered_products = []

        for product in list_product_element:
            try:
                if self._check_condition_with_llm(product, condition):
                    filtered_products.append(product)
            except Exception as e:
                # Log ou passer en silence
                continue

        return filtered_products


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem and a pd.dataframe corresponding to the recommended product if necessary"
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"},
              "structured_product": {"type": "object",
                                     "description": "optional products recommended in a structured format",
                                     "nullable": True},}
    output_type = "any"

    def forward(self, answer: Any, structured_product: Optional[object] = None) -> (Any, Any):
        return answer, structured_product