from textwrap import dedent

from langchain.prompts import PromptTemplate

prompt_template_rag_mc = PromptTemplate(
    input_variables=["statements", "reference"],
    template=dedent(
        """<|begin_of_text|>
        
        <|start_header_id|>system<|end_header_id|>

            You are an expert AI tasked with identifying the single correct statement from a list of four options based on a provided reference text. Follow these guidelines:
            - Carefully analyze the reference text to ensure full understanding of its details, context, and meaning.
            - Compare each of the four statements to the reference text. Evaluate their accuracy, relevance, and adherence to the reference content.
            - Select the one statement (from options 1, 2, 3, or 4) that has the highest likelihood of being correct based on the reference text.
            - Return only the number corresponding to the most likely statement. 
            - Do not provide explanations, additional context, or any other output.
            - There will always be one correct statement and three incorrect statements.
            - If none of the statements are correct, select option the option that is closest to being correct.
            - Always consider the reference text as the ultimate source of truth.
            - If you are unsure about the correct answer, make your best judgment based on the information provided.
            - Always provide an answer, even if you are uncertain about its accuracy.
        
            Answer in JSON format.
            Do not include any reasoning or additional information in the response.
            The response should only contain the final JSON string.
            Use double quotes for strings.
            Use correct JSON syntax.
            The JSON should contain the following fields:
            - correct_statement: The number of the correct statement (1, 2, 3, or 4).
            - reasoning: Briefly explain why you selected the statement as the correct one.

            
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
            Reference: 
            "The American Civil War began in 1861 and ended in 1865. It was fought primarily over issues of slavery and states' rights."
            
            Statements:
            1. The American Civil War began in 1861 and ended in 1865.
            2. The American Civil War was fought between 1860 and 1864.
            3. The American Civil War was primarily about taxation and trade.
            4. The American Civil War lasted for ten years.
        
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            {{
                "correct_statement": 1,
                "reasoning": "The statement accurately reflects the timeline of the American Civil War."
            }}

        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>

            Reference: 
            Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water. The process primarily produces glucose and oxygen.
            
            Statements:
            1. Photosynthesis occurs in plants and produces nitrogen and carbon dioxide.
            2. Green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen through photosynthesis.
            3. Photosynthesis requires oxygen to create glucose and water.
            4. Photosynthesis is a process unique to animals and fungi.    
        
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|> 
            
            {{
                "correct_statement": 2,
                "reasoning": "The statement accurately describes the process of photosynthesis in green plants."
            }}
         
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
            
            Reference:
            {reference}
            
            Statements:
            {statements}
                
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
            <|end_of_text|> 
        """
    ),
)
