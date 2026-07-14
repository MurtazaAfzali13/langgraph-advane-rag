from typing import TypedDict,List


class GraphState(TypedDict):
    """
    Represent the state of graph
    
    Attributes:
    question:question
    generation: LLM generation
    web_search: wheter to add search 
    documents:List of documents
    """
    
    question:str
    generation:str
    web_search:bool
    documents:List[str]