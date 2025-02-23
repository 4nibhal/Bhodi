"""
Module bhodi_doc_analyzer.workflow

This module builds and compiles the workflow for processing conversation context
and generating responses.
"""

from typing import List, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser


from bhodi_doc_analyzer.config import retriever, llm, reranker, sequencer
from bhodi_doc_analyzer.utils import count_tokens, fast_tokenize, save_log
from indexer.config_indexer import persistent_retriever

# =============================================================================
# AGENT STATE AND RESPONSE PARSER SETUP
# =============================================================================
class AgentState(TypedDict):
    """
    Typed dictionary representing the agent's state.
    
    Attributes:
        messages (List[dict]): List of messages in the conversation.
        context (str): The conversation context.
        input (str): The user input.
    """
    messages: List
    context: str
    input: str

class AssistantAnswer(BaseModel):
    """
    Structured schema for the assistant's answer.
    """
    answer: str = Field(..., description="The assistant's answer content")

# Initialize the output parser using our schema.
answer_parser = PydanticOutputParser(pydantic_object=AssistantAnswer)

# =============================================================================
# PROMPT AND RESPONSE PROCESSING FUNCTIONS
# =============================================================================
def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieves context by querying both the volatile and persistent vectorstores,
    then applying the sequencer 
    to refine and filter the combined documents.
    """
    query = state['input']
    # Query the volatile retriever.
    volatile_docs = retriever.invoke(query)
    # Query the persistent retriever.
    persistent_docs = persistent_retriever.invoke(query)
    # Combine documents from both sources.
    all_docs = volatile_docs + persistent_docs    
    # Apply the sequencer: rerank and further refine documents.
    sequenced_docs = sequence_documents(query, all_docs)
    
    context = "\n".join([doc.page_content for doc in sequenced_docs])
    if count_tokens(context) > 1000:
        context = context[:1000]
    return {"context": context}

def generate_response(state: AgentState) -> AgentState:
    """
    Generates an LLM response using the refined prompt and conversation context.
    The response is then parsed using the structured output parser, with a fallback
    to the raw response.
    """
    prompt_text = refine_prompt(state.get('context', ''), state['input'])
    _ = fast_tokenize(prompt_text)
    # Transform state messages into valid message objects.
    transformed_messages = [transform_message(msg) for msg in state['messages']]
    # Append the current prompt as a new human message.
    messages = transformed_messages + [HumanMessage(content=prompt_text)]
    raw_response = llm.invoke(messages)
    save_log(f"Raw response: {raw_response}")


    try:
        structured_obj = answer_parser.parse(raw_response)
        answer_text = structured_obj.answer
    except ValueError as e:
        save_log(f"Parsing error: {e}")
        answer_text = raw_response  # Fallback to raw response

    
    # Ensure answer_text is a plain string.
    if not isinstance(answer_text, str):
        try:
            answer_text = answer_text.content
        except AttributeError:
            answer_text = str(answer_text)
    
    return {"answer": answer_text}

def summarize_text(text: str) -> str:
    """
    Generates a summary for long texts using a dedicated summarization model (sequencer).
    
    If the text is not too long, it is returned unchanged.
    """
    # Define el umbral a partir del cual se realiza el resumen
    if len(text) < 2500:
        return text
    # Llama al pipeline de summarization (sequencer)
    summary = sequencer(text, max_length=1500, min_length=500, truncation=True)
    return summary[0]['summary_text']

def refine_prompt(context: str, user_input: str) -> str:
    """
    Refines the prompt by including context and the user input.
    """
    if count_tokens(context) > 1200:
        context = summarize_text(context)
    prompt = f"Context: {context}\nQuestion: {user_input}"
    print(f"Prompt tokens: {count_tokens(prompt)}")
    return prompt

def transform_message(msg: AgentState):
    """
    Transforms a message dictionary into a valid message object acceptable by the LLM.
    """
    role = msg.get("role")
    content = msg.get("content", "")
    if role in ["question", "human", "user"]:
        return HumanMessage(content=content)
    if role in ["answer", "assistant", "ai"]:
        return AIMessage(content=content)
    # Fallback: use human message
    return HumanMessage(content=content)

def rerank_documents(query: str, docs: List[Any]) -> List[Any]:
    """
    Reranks documents based on the similarity between the query and each document
    using a Hugging Face reranker.
    
    Args:
        query (str): The user query.
        docs (List[Any]): List of documents (each with 'page_content' attribute).
    
    Returns:
        List[Any]: Documents reranked in descending order of relevance.
    """
    scored_docs = []
    for doc in docs:
        result = reranker(
            (query, doc.page_content),
            padding=True,
            truncation=True,
            max_length=512
        )
        # Check if result is a list or a dict.
        score = result[0]["score"] if isinstance(result, list) else result["score"]
        scored_docs.append((score, doc))
    ranked_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
    return ranked_docs

def sequence_documents(query: str, docs: List[Any]) -> List[Any]:
    """
    Applies a series of steps to refine the ranking and content of retrieved documents.
    
    1. Applies the Hugging Face reranker to order documents.
    2. Optionally summarizes documents that are too long.
    
    Args:
        query (str): The user query.
        docs (List[Any]): List of documents (each with 'page_content' attribute).
        
    Returns:
        List[Any]: Refined list of documents.
    """
    # Step 1: Rerank the documents.
    reranked_docs = rerank_documents(query, docs)    
    # Step 2: Optionally, apply summarization to each document if content is too long.
    # Here we assume each document is an object with attribute 'page_content'
    #  and we update it with a summary.
    for doc in reranked_docs:
        if count_tokens(doc.page_content) > 300:
            # Replace content with a summary (could use a dedicated summarization model)
            doc.page_content = summarize_text(doc.page_content)
    return reranked_docs

# =============================================================================
# BUILDING THE WORKFLOW WITH LANGGRAPH (without persistent MemorySaver)
# =============================================================================
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the workflow without a checkpointer.
compiled_graph = workflow.compile()


def execute_workflow(state: AgentState) -> AgentState:
    """
    Executes the compiled workflow by running the 'retrieve' node
    and then the 'generate' node, updating the state accordingly.
    """
    state.update(retrieve_context(state))
    state.update(generate_response(state))
    return state


class CallableStateGraph:
    """
    A wrapper that makes a compiled state graph callable.
    """
    def __init__(self, graph, executor):
        self.graph = graph
        self.executor = executor

    def __call__(self, state: AgentState) -> AgentState:
        return self.executor(state)


# Wrap the compiled graph with our callable wrapper using execute_workflow.
graph = CallableStateGraph(compiled_graph, execute_workflow)