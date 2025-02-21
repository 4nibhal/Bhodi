from langchain_core.messages import HumanMessage
from bhodi_doc_analyzer.config import llm
from bhodi_doc_analyzer.workflow import answer_parser
from bhodi_doc_analyzer.utils import save_log

def generate_chat_response(tech_prompt: str) -> str:
    """
    Generates a chat response from the language model using the provided technical prompt.
    This function is executed synchronously and is designed to be called from a separate thread.

    Args:
        tech_prompt (str): The technical prompt for the language model.

    Returns:
        str: The generated answer from the language model.
    """
    raw_response: str = llm.invoke([HumanMessage(tech_prompt)])
    save_log(f"Raw response (structured mode): {raw_response}")
    try:
        structured = answer_parser.parse(raw_response)
        answer_text: str = structured.answer
    except ValueError as error:  # Catching a more specific exception
        save_log(f"Parsing error (structured mode): {error}")
        answer_text = raw_response  # Fallback to raw response if parsing fails
    return answer_text
