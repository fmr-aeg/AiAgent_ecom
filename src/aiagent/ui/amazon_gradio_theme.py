from gradio.themes.base import Base
from gradio.themes.utils import colors


class AmazonTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.orange,
            secondary_hue=colors.gray,
            font=["Arial", "sans-serif"],
            font_mono=["Courier New", "monospace"]
        )

        self.set(
            body_background_fill="#f3f3f3",
            border_color_primary="#ddd",
            shadow_drop="0 2px 4px rgba(0,0,0,0.08)",
            button_primary_background_fill="#34455c",
            button_primary_text_color="#ffffff",
            button_primary_background_fill_hover="#f5aa3b",

            button_secondary_background_fill="#f5aa3b",
            button_secondary_background_fill_hover="#de952a",
            button_secondary_border_color="#ccc",
            button_secondary_text_color="#473a25",

            button_border_width='0px',
            # button_radius='8px',

            # Champs de saisie (input)
            input_background_fill="#fffff",
            input_border_color="#ccc",
            input_border_width="1px",
            input_radius="6px",
            input_shadow="0 1px 2px rgba(0, 0, 0, 0.1)",
            input_padding="10px 14px",
            input_text_size="16px",
            # input_text_align = "left",

            # Liens
            link_text_color="#007185",
            link_text_color_hover="#C7511F"
        )
