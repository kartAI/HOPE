from textwrap import dedent

from langchain.prompts import PromptTemplate


prompt_template_rag = PromptTemplate(
    input_variables=["question", "reference"],
    template=dedent(
        """        
            Answer the following question using the provided context.
            Keep your answer concise and relevant.
            Do not include any irrelevant information.
            Do not include any reasoning or additional information in the response.
            Only provide the answer to the question.
            Follow the format of the example provided.

            ## Example 1:
            Reference: Paris is the capital of France. Berlin is the capital of Germany. Rome is the capital of Italy.
            Question: What is the capital of France?
            Answer: Paris
            
            ## Example 2:
            Reference: The United States is the third largest country by land area. The US has a population of over 331 million people. The US is known for its diverse culture and is home to people from all over the world.
            Question: What is the population of USA?
            Answer: 331 million people
            
            ## End of example
            
            ## Question 
            {question}
                
            ## Context:
            {reference}
            
            ### Instructions:
            Do not include any reasoning or additional information in the response.
            Only provide the answer to the question.
            
        """
    ),
)
