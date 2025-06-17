from textwrap import dedent

from langchain.prompts import PromptTemplate


_json_formatting = dedent(
    """
    Answer in JSON format.
    Do not include any reasoning or additional information in the response.
    The response should only contain the final JSON string.
    Use double quotes for strings.
    Use correct JSON syntax.
    """
)


prompt_template_qna_generator = PromptTemplate(
    input_variables=["number_of_questions", "reference_text"],
    template=dedent(
        """
        You are an AI that generates questions from a given text. 
        Please generate {number_of_questions} questions and answers based on the following reference text. 
        The questions should be relevant to the text.
        The questions should only be based on the information provided in the text.
        Answering the questions should only be possible with the provided reference text. 

        """
        + _json_formatting
        + """

        ## Reference text:
        {reference_text}
        
        ## Example input:
        {{
            "reference_text": "Paris is the capital of France. Berlin is the capital of Germany. Rome is the capital of Italy.",
            "number_of_questions": 3
        }}
        
        ## Example output:
        {{
            "questions": [
                {{
                    "question": "What is the capital of France?",
                    "answer": "Paris"
                }},
                {{
                    "question": "What is the capital of Germany?",
                    "answer": "Berlin"
                }},
                {{
                    "question": "What is the capital of Italy?",
                    "answer": "Rome"
                }}
            ]
        }}
        """
    ),
)


prompt_template_qna_critic = PromptTemplate(
    input_variables=["question", "reference_text"],
    template=dedent(
        """
            Evaluate weather the following question is relevant to the given text.
            Please evaluate the following question based on the following reference text:
            
        """
        + _json_formatting
        + """
              

        ## Example input:
        {{
            "reference_text": "Paris is the capital of France. Berlin is the capital of Germany. Rome is the capital of Italy.",
            "question": "What is the capital of France?"
        }}
        
        ## Example output:
        {{
            "is_relevant": true,
            "reason": "The capital of France is mentioned in the reference text."
        }}

            
        ### Question:
        {question}
        
        ### Reference text:
        {reference_text}  

        """
    ),
)
