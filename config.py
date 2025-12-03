import os
from dotenv import load_dotenv

load_dotenv()

def load_openai_api_key():
    """Load OpenAI API key from .env"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file")
    return OPENAI_API_KEY


def load_claude_api_key():
    """Load Claude (Anthropic) API key from .env"""
    CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        raise ValueError("Missing ANTHROPIC_API_KEY in .env file")
    return CLAUDE_API_KEY


def load_langsmith_api_key():
    """Load LangSmith API key from .env"""
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

    if not LANGSMITH_API_KEY:
        print("Warning: LANGSMITH_API_KEY not found in .env file")
        return None

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "research-agent-langgraph")

    return LANGSMITH_API_KEY
