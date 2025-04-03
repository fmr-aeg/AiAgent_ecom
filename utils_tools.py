from smolagents import tool, Tool
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
from io import BytesIO


class ParserProductDescriptionWithoutGuideTool(Tool):
    name = "parse_product_description_without_guide"
    description = "tool that extract every relevant information from product description provided. It determine which are the best feature to find for a given product and return a structured response using json format"
    inputs = {"product_description": {"type": "string",
                                      "description": "The product description containing every information on the product"},
              }
    output_type = "string"

    def __init__(self, model,
                 system_prompt,
                 **kwargs):
        super().__init__( **kwargs)
        self.model = model
        self.system_prompt = system_prompt

    def _preprocessing_message(self, product_description):
        messages = [{"role": "system",
                     "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user",
                     "content": [{"type": "text", "text": product_description}]}
                    ]
        return messages

    def forward(self, product_description: str):
        messages = self._preprocessing_message(product_description)

        return self.model(messages, response_format={"type": "json_object"}).content


class ParserProductDescriptionWithGuideTool(Tool):
    name = "parse_product_description_with_guide"
    description = "tool that retrieve on a product description the value for a given list of relevant feature. It return a structured response using json format"
    inputs = {"product_description": {"type": "string",
                                      "description": "The product description containing every information on the product"},
              "product_feature": {"type": "array",
                                  "description": "The list of feature that should be retrieve from product description"},
              }
    output_type = "string"

    def __init__(self, model,
                 system_prompt,
                 **kwargs):
        super().__init__( **kwargs)
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

        return self.model(messages, response_format={"type": "json_object"}).content


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
        "https": "scraperapi.country_code=us:b3113d8cc78c7f441f071729e15055e1@proxy-server.scraperapi.com:8001"
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
def get_raw_description_from_product_url(product_url: str) -> str:
    """
    Extract a raw string containing all the information available on the product on Amazon

    Args:
        product_url: The url of the product sell on Amazon.com

    Returns:
        A string containing all the description found on the product page
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(product_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        preprocessed_html = soup.title.string if soup.title else 'Title not found'
        preprocessed_html += "\n"

        # Extraction de la description
        product_description = soup.find("div", id="productDescription")
        alt_description = soup.find("div", id="feature-bullets")
        seller_description = soup.find("div", id="aplus")
        price = soup.find('span', class_='a-offscreen')

        if product_description:
            preprocessed_html += "## product description by Website## \n\n"
            preprocessed_html += product_description.get_text(strip=True) + '\n\n'

        if alt_description:
            preprocessed_html += "## additional description ## \n\n"
            preprocessed_html += alt_description.get_text(strip=True) + '\n\n'

        if seller_description:
            preprocessed_html += "## product description by Seller ## \n\n"
            preprocessed_html += seller_description.get_text(strip=True)

        return preprocessed_html

    except requests.exceptions.RequestException as e:
        return f"Error : {str(e)}"


@tool
def make_a_search_on_amazon(keyword: str) -> list[dict]:
    """
    function to retrieve a list of products resulting from a search on the amazon search engine

    Args:
        keyword: the keyword to search for in the search engine

    Returns:
        a list containing one json per product with the following three keys :
            - title : title of the product
            - image_url : url of the product's image
            - product_link : url of the product page
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    url = f"https://www.amazon.com/s?k={keyword.replace(' ', '+')}"

    # Envoi de la requête HTTP à Amazon
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Erreur lors de la récupération de la page")

    # Utilisation de BeautifulSoup pour analyser le HTML de la page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Récupérer les 10 premiers produits
    products = []
    product_elements = soup.find_all('div', role='listitem')

    for product in product_elements[:10]:  # Limiter à 10 produits
        tag_sponsorised = product.find('span', class_='puis-label-popover-default')

        if tag_sponsorised:
            continue

        title_element = product.find('h2')
        image_element = product.find('img', class_='s-image')
        link_element = product.find('a', class_='a-link-normal')

        if title_element and image_element and link_element:
            title = title_element.get_text()
            image_url = image_element['src']
            product_link = 'https://www.amazon.com' + link_element['href']

            # Ajouter les données du produit dans la liste
            products.append({
                'title': title,
                'image_url': image_url,
                'product_link': product_link
            })

    return products

@tool
def compare_products(features: list[str], products: list[any]) -> pd.DataFrame:
    """
    Compare products based on specified features and return a dataframe to easily read the comparisons

    Args:
        features: List of features to compare (e.g., ['Price', 'Screen', 'Processor'])
        products: List of products as dictionaries (e.g., [{'Name': 'Product 1', 'Price': 500, 'Screen': '15"', 'Processor': 'Intel i5'}, {'Name': 'Product 2', 'Price': 600, 'Screen': '17"', 'Processor': 'Intel i7'}])

    Returns:
        A pandas DataFrame representing the comparison of the products
    """

    # Créer une liste pour les lignes du tableau
    comparison_data = []

    # Ajouter les produits dans les colonnes du tableau
    for product in products:
        # Extraire uniquement les caractéristiques pertinentes pour ce produit
        product_comparison = {feature: product.get(feature, 'N/A') for feature in features}
        product_comparison['Name'] = product.get('Name', 'unknown')  # Ajouter le nom du produit comme première colonne
        comparison_data.append(product_comparison)

    # Créer un DataFrame pandas à partir de la comparaison
    df_comparison = pd.DataFrame(comparison_data, columns=features)

    return df_comparison

@tool
def compare_products_bis(products: list[any]) -> pd.DataFrame:
    """
    Compare products based on specified features and return a dataframe to easily read the comparisons

    Args:
        products: List of products as dictionaries (e.g., [{'Name': 'Product 1', 'Price': 500, 'Screen': '15"', 'Processor': 'Intel i5'}, {'Name': 'Product 2', 'Price': 600, 'Screen': '17"', 'Processor': 'Intel i7'}])

    Returns:
        A pandas DataFrame representing the comparison of the products
    """

    df_comparison = pd.DataFrame(products)

    return df_comparison

@tool
def display_product_image(product_url: str) -> Image:
    """
    Download the product image from product_url and return a PIL.Image object to display the product

    Args:
        product_url: the url of the image to be downloaded
    Returns:
        A image in format PIL.Image downloaded from url
    """
    response = requests.get(product_url)

    if response.status_code != 200:
        print("Error on downloading image process")

    image = Image.open(BytesIO(response.content))
    return image