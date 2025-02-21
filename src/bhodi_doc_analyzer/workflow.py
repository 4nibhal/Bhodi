from typing import List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser


from bhodi_doc_analyzer.config import retriever, llm
from bhodi_doc_analyzer.utils import count_tokens, fast_tokenize, save_log
from indexer.config_indexer import persistent_retriever

# =============================================================================
# AGENT STATE AND RESPONSE PARSER SETUP
# =============================================================================
class AgentState(TypedDict):
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
def retrieve_context(state: AgentState) -> dict:
    """
    Retrieve the conversation context from both the volatile (chat memory)
    and the persistent (document index) vectorstores.
    
    The function queries both retrievers, concatenates their documents, and
    returns a trimmed context.
    """
    query = state['input']
    # Query the volatile retriever (from chat config).
    volatile_docs = retriever.invoke(query)
    # Query the persistent retriever (from document indexing).
    persistent_docs = persistent_retriever.invoke(query)
    # Combine documents from both sources.
    all_docs = volatile_docs + persistent_docs
    context = "\n".join([doc.page_content for doc in all_docs])
    if count_tokens(context) > 1000:
        context = context[:1000]
    return {"context": context}

def summarize_text(text: str) -> str:
    return text if len(text) < 1000 else text[:1000] + "..."

def refine_prompt(context: str, user_input: str) -> str:
    """
    Refines the prompt by including context and the user input.
    """
    if count_tokens(context) > 1200:
        context = summarize_text(context)
    prompt = f"Context: {context}\nQuestion: {user_input}"
    print(f"Prompt tokens: {count_tokens(prompt)}")
    return prompt

def transform_message(msg: dict):
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

def generate_response(state: AgentState) -> dict:
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
    except Exception as e:
        save_log(f"Parsing error: {e}")
        answer_text = raw_response  # Fallback to raw response

    
    # Ensure answer_text is a plain string.
    if not isinstance(answer_text, str):
        try:
            answer_text = answer_text.content
        except AttributeError:
            answer_text = str(answer_text)
    
    return {"answer": answer_text}

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
graph = workflow.compile()
