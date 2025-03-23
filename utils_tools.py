from smolagents import tool
import requests
from bs4 import BeautifulSoup


@tool
def get_amazon_product_description(product_url: str) -> str:
    """
    Extract a complete product description from a product on Amazon.com

    Args:
        product_url: The url of the product sell on Amazon.com

    Returns:
        A string containing every information on the product
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(product_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        preprocessed_html = soup.title.string if soup.title else 'Aucun titre trouvé'
        preprocessed_html += "\n"

        # Extraction de la description
        product_description = soup.find("div", id="productDescription")
        alt_description = soup.find("div", id="feature-bullets")
        seller_description = soup.find("div", id="aplus")

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


