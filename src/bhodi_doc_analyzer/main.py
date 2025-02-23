"""
Module bhodi_doc_analyzer.main

This module serves as the entry point for the Bhodi chat application.
It instantiates the ChatApp class and starts the chat interface.
"""

from bhodi_doc_analyzer.assistant import ChatApp

def main_menu():
    """
    Entry point for the Bhodi chat application.

    This function creates an instance of ChatApp and runs the application.
    """
    app = ChatApp()
    app.run()

if __name__ == "__main__":
    main_menu()
