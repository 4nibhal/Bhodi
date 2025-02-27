"""
Module bhodi_doc_analyzer.assistant

This module implements the chat application with a textual user interface for Bhodi.
It includes the classes and functions required to:
  - Configure and manage the appearance and interaction of the interface.
  - Display user messages and generated responses.
  - Integrate the processing workflow for conversations via a StateGraph that retrieves context
    and generates responses using the language model (LLM).

Custom components defined in this module include:
  - FocusableContainer: a focusable container widget for the conversation.
  - MessageBox: a widget for displaying individual messages.
  - ChatApp: the main application that orchestrates input, output, and conversation updates.
"""

import asyncio
from typing import List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Markdown

# Importing the compiled workflow graph from workflow.py.
from bhodi_doc_analyzer.workflow import graph, AgentState

# Import the volatile (in-memory) vectorstore and the persistent one.
from bhodi_doc_analyzer.config import vectorstore as volatile_vectorstore
from indexer.config_indexer import persistent_vectorstore

class FocusableContainer(Container):
    """A container widget that is focusable."""
    pass

class MessageBox(Widget):
    """
    A widget for displaying a single message in the conversation.
    
    Attributes:
        text (str): The content of the message.
        role (str): The role of the message (e.g., 'question' or 'answer').
    """
    def __init__(self, text: str, role: str) -> None:
        self.text: str = text
        self.role: str = role
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Markdown(self.text, classes=f"message {self.role}")

class ChatApp(App):
    """
    Chat application implemented with a textual user interface.
    """
    TITLE: str = "chatui"
    SUB_TITLE: str = "ChatGPT directly in your terminal"
    CSS_PATH: str = "static/styles.css"
    BINDINGS = [
        Binding("q", "quit", "Quit", key_display="Q / CTRL+C"),
        ("ctrl+x", "clear", "Clear"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize conversation history as a list of message dictionaries.
        self.conversation_history: List[AgentState[str, str]] = []

    def compose(self) -> ComposeResult:
        """
        Defines the layout of the chat application.
        """
        yield Header()
        with FocusableContainer(id="conversation_box"):
            welcome_message: str = (
                "Welcome to the assistant!\n"
                "Type your question, press Enter or click the 'Send' button,\n"
                "and wait for the response.\n"
                "Below you will find some useful commands."
            )
            yield MessageBox(welcome_message, role="info")
        with Horizontal(id="input_box"):
            yield Input(placeholder="Type your query", id="message_input")
            yield Button(label="Send", variant="success", id="send_button")
        yield Footer()

    def on_mount(self) -> None:
        """
        Called when the application is mounted. Sets focus to the input field.
        """
        self.query_one(Input).focus()

    def action_clear(self) -> None:
        """
        Clears the conversation area by removing all existing message widgets.
        """
        conversation_box = self.query_one("#conversation_box")
        for child in list(conversation_box.children):
            child.remove()

    async def on_button_pressed(self) -> None:
        """
        Handler for the 'Send' button press. Initiates processing of the conversation.
        """
        await self.process_conversation()

    async def on_input_submitted(self) -> None:
        """
        Handler for input submission. Initiates processing of the conversation.
        """
        await self.process_conversation()

    async def process_conversation(self) -> None:
        """
        Processes a single conversation exchange by:
          - Capturing user input and updating the conversation history.
          - Building the agent state and executing the workflow via a StateGraph.
          - Updating the UI with the generated answer.
          - Updating both the volatile and persistent vectorstores with new messages.
        """
        message_input: Input = self.query_one("#message_input", Input)
        if message_input.value == "":
            return

        button: Button = self.query_one("#send_button")
        conversation_box = self.query_one("#conversation_box")
        self.toggle_widgets(message_input, button)

        # Capture the user's message and update the conversation history.
        user_message: str = message_input.value
        self.conversation_history.append({"role": "question", "content": user_message})
        conversation_box.mount(MessageBox(user_message, "question"))
        conversation_box.scroll_end(animate=False)

        with message_input.prevent(Input.Changed):
            message_input.value = ""

        # Build agent state.
        agent_state: AgentState = {
            "messages": self.conversation_history,
            "input": user_message,
            "context": ""  # initial context value
        }

        # Invoke the workflow (StateGraph) asynchronously using asyncio.to_thread because
        # the graph.run method is synchronous.
        result: AgentState = await asyncio.to_thread(graph, agent_state)
        answer_text: str = result.get("answer", "No response")

        # Append the answer to the conversation history and update the UI.
        self.conversation_history.append({"role": "answer", "content": answer_text})
        conversation_box.mount(MessageBox(answer_text, "answer"))
        self.toggle_widgets(message_input, button)
        conversation_box.scroll_end(animate=False)

        # Update both vectorstores with the new conversation messages.
        volatile_vectorstore.add_texts([user_message, answer_text])
        persistent_vectorstore.add_texts([user_message, answer_text])

    def toggle_widgets(self, *widgets: Widget) -> None:
        """
        Toggles the 'disabled' state of the provided widgets.
        
        Args:
            widgets (Widget): The widgets to be toggled.
        """
        for widget in widgets:
            widget.disabled = not widget.disabled

if __name__ == "__main__":
    ChatApp().run()