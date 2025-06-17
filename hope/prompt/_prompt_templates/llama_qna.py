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


#     Follow the following format schema:
# """
# + str(SchemaQnaGenerator.model_json_schema())
# + """

prompt_template_qna_generator = PromptTemplate(
    input_variables=["number_of_questions", "reference_text"],
    template=dedent(
        """<|begin_of_text|>
        
        <|start_header_id|>system<|end_header_id|>
        
            You are a highly capable question-answering system designed to generate relevant questions and their corresponding answers from a given text snippet. 
            Your goal is to create questions that are tightly related to the content of the snippet, ensuring that each question is answerable by the information in the text. 
            These questions will later be used in a Retrieval-Augmented Generation (RAG) setup, so it is critical that the questions are specific enough to allow the correct text snippet to be retrieved through a dense vector search.

            Instructions:

            - Question Relevance: Each question must be directly related to the content of the provided text. Avoid asking general or unrelated questions.
            - Specificity: The questions should contain enough specific details (names, dates, locations, concepts, etc.) so that when the corresponding answer is provided, it can be easily linked back to the original text through a vector search.
            - Clarity and Precision: Make sure the questions are clear, precise, and unambiguous. Do not use overly broad or vague questions.
            - Question Variety: Generate a variety of question types, such as:
            - Fact-based questions: What, Who, Where, When, etc.
            - Reasoning-based questions: Why, How, etc.
            - Answer Context: Provide the correct, concise answer to each question using information directly from the text snippet. Avoid introducing external knowledge or speculative answers.

            Guidelines for the LLM:

            - Focus on extracting important details from the snippet, making sure that each question touches on a specific, retrievable fact.
            - Keep the answers concise, using exact wording from the snippet when possible, as this ensures the correct context is maintained.
            - Ensure the questions are structured in such a way that vector search algorithms will easily find the snippet as the answer source.


        """
        + _json_formatting
        + """

        
        
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>

            Number of questions: 3
            
            Reference:
            "The Eiffel Tower was completed in 1889 and is located in Paris, France. It was designed by Gustave Eiffel and stands 324 meters tall."

        <|eot_id|> 

        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
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
            
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
            
            Number of questions: {number_of_questions}
            
            Reference:
            {reference_text}
            
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            <|end_of_text|> 

        """
    ),
)

#     Follow the following format schema:
# """
# + str(SchemaQnaCritic.model_json_schema())
# + """
prompt_template_qna_critic = PromptTemplate(
    input_variables=["question", "reference_text"],
    template=dedent(
        """<|begin_of_text|>
        
        <|start_header_id|>system<|end_header_id|>

            You are an AI that evaluates questions based on a given text.
            Evaluate weather the following question is relevant to the given text.
            Please evaluate the following question based on the following reference text:
            
        """
        + _json_formatting
        + """
        

        
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
            Reference text:
            "Paris is the capital of France. Berlin is the capital of Germany. Rome is the capital of Italy."
            
            Question:
            "What is the capital of France?"
        
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        {{
            "is_relevant": true,
            "reason": "The capital of France is mentioned in the reference text."
        }}
        
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
            Reference text:
            {reference_text}
            
            Question:
            {question}

        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            <|end_of_text|> 
        """
    ),
)
