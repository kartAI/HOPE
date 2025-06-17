from textwrap import dedent
from json import dumps

from langchain.prompts import PromptTemplate
from hope.prompt import SchemaMultipleChoice


_json_formatting = dedent(
    f"""
    Answer in JSON format.
    Do not include any reasoning or additional information in the response.
    The response should only contain the final JSON string.
    Use double quotes for strings.
    Use correct JSON syntax.
    """
    # Follow the following format schema:
    # {dumps(SchemaMultipleChoice.model_json_schema()).replace(r"{", r'{{').replace(r"}", r'}}')}
    # """
)


prompt_template_mc_generator = PromptTemplate(
    input_variables=["reference_text"],
    template=dedent(
        """<|begin_of_text|>
        
        <|start_header_id|>system<|end_header_id|>
        
            You are an advanced language model tasked with generating benchmark test cases for a Retrieval-Augmented Generation (RAG) system. 
            Your job is to produce four text statements based on a given text snippet:

            - Statement 1 must be directly derived from the text snippet. It should convey factual information from the snippet with clarity and precision.
            - Statements 2, 3, and 4 must not be derived from the text snippet. However, they should remain within the same thematic domain (e.g., topic, field, or subject) as the snippet. These statements must be semantically different from each other and Statement 1 but should appear plausible within the domain.
            
            Rules:
            - Relevance: Ensure that all four statements are relevant to the broader domain of the text snippet. Avoid introducing unrelated topics.
            - Semantic Variation: Each statement should differ in phrasing, focus, or perspective to test the system's ability to retrieve specific contextual matches.
            - Correctness: Statements 2, 3, and 4 should be factually incorrect or unsupported by the snippet, while Statement 1 is an accurate reflection of the provided snippet.

        """
        + _json_formatting
        + """

        
        
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
            Reference:
            "Solar energy is a renewable energy source that reduces greenhouse gas emissions, decreases dependence on fossil fuels, and has low operating costs after installation."

        <|eot_id|> 

        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            {{
                "derived_statement": "Solar energy is a renewable source that helps lower greenhouse gas emissions and reduces reliance on fossil fuels.",
                "false_statements": [
                    "Solar energy systems have higher greenhouse gas emissions during operation than fossil fuels.",
                    "Wind energy is the most cost-effective renewable energy source, surpassing solar energy in affordability.",
                    "Solar panels rely heavily on non-renewable rare-earth materials, making them unsustainable in the long term."
                ]
            }}
            
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
            
            Reference:
            {reference_text}
            
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            <|end_of_text|> 

        """
    ),
)
