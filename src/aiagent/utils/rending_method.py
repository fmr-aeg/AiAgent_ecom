from litellm import completion


class cleaning_model_thinking:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.system_message = [{
            "role": "system",
            "content": ("You are an assistant that helps explain the reasoning or actions of an AI agent to end users. "
                        "You translate either 'Thought' steps or blocks of code into clear, natural language that a "
                        "non-technical user can understand. "
                        "Each response must be a **single, precise, user-facing sentence** that keeps the **key "
                        "intent of the block**. "
                        "Avoid vague summaries like 'help you decide' or 'comparing products' — be specific about "
                        "what the AI is doing."
                        "Always speaks in the first person singular")
        }]

    @staticmethod
    def formalize_message_block(thinking_bloc: str):
        message = [{
            "role": "user",
            "content": (f"Here is a block from an AI agent that helps users compare products on Amazon. "
                        f"Please summarize exactly what the AI is doing in one short and clear sentence, without "
                        f"using any technical terms or code.\n\n "
                        f"Block:\n:\n\n {thinking_bloc}")
        }]

        return message

    def __call__(self, thinking_bloc):
        user_message = self.formalize_message_block(thinking_bloc)

        res = completion(
            model=self.model_name,
            messages=self.system_message + user_message)

        return res.choices[0].message.content
