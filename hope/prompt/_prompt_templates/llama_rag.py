from textwrap import dedent

from langchain.prompts import PromptTemplate


prompt_template_rag = PromptTemplate(
    input_variables=["question", "reference"],
    template=dedent(
        """<|begin_of_text|>
        
        <|start_header_id|>system<|end_header_id|>

            Answer the following question using the provided context.
            Keep your answer concise and relevant.
            Do not include any irrelevant information.
            Do not include any reasoning or additional information in the response.
            Only provide the answer to the question.
            Follow the format conversation.
            Do not include any reasoning or additional information in the response.
            Only provide the answer to the question.
            Let's think step by step. 
            
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
            Reference: 
            Paris is the capital of France. Berlin is the capital of Germany. Rome is the capital of Italy.
            
            Question: 
            What is the capital of France?
        
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            Paris

        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>

            Reference: 
            The United States is the third largest country by land area. The US has a population of over 331 million people. The US is known for its diverse culture and is home to people from all over the world.

            Question: 
            What is the population of USA?
        
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|> 
            
            331 million people
         
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
            
            Reference:
            {reference}
            
            Question:
            {question}
                
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            <|end_of_text|> 
        """
    ),
)
