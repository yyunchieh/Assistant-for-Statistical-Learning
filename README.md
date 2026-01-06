# Statistical Learning RAG Assistant

A configuration-driven RAG (Retrieval-Augmented Generation) system built with LangGraph for answering questions about The Elements of Statistical Learning.
The project emphasizes reproducibility, agent-based orchestration, and evaluation of hallucination behavior.

## Features

- **PDF-based RAG**: Query the "Elements of Statistical Learning" textbook
- **Source Citations**: Every answer includes page numbers and relevant excerpts
- **LangSmith Integration**: Full tracing and monitoring of all operations
- **LLM-as-a-Judge Evaluation**: Automated hallucination detection
- **Streamlit UI**: User-friendly web interface
- **Markdown Export**: Download Q&A with sources as markdown files

## Architecture

### Components
```
├── main.py               # Streamlit UI (user interaction & input handling)
├── graph.py              # LangGraph-based RAG workflow (agent graph definition)
├── langgraph.json        # Declarative graph configuration & entry point
├── config.py             # Centralized environment & API configuration
├── .env                  # Environment variables
└── the Elements of Statistical Learning.pdf   # Knowledge source
```

### Workflow
```
User Query
   ↓
Streamlit UI (main.py)
   ↓
LangGraph Runtime
   ├── Graph Entry (langgraph.json)
   └── Agent Graph (graph.py)
           ↓
     Vector Search
           ↓
     Context Retrieval
           ↓
     LLM Generation
           ↓
 Answer + Sources
           ↓
 Page Numbers + Excerpts
```

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key
- LangSmith API key (for monitoring)

### Setup

1. **Clone or download this project**

```bash
git clone https://github.com/yyunchieh/Assistant-for-Statistical-Learning

cd Assistant-for-Statistical-Learning
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Sign up for LangSmith**
   
- Create an account at [Langsmith](https://smith.langchain.com)
- Generate your API Key
  
4. **Create `.env` file:**

```bash
OPENAI_API_KEY=your-openai-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RAG_assistant_evaluator
```

5. **Place the PDF:**

- Download **The Elements of Statistical Learning PDF** from [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7)



## Evaluation Setup (LLM-as-a-Judge)
This project used custom LangSmith Evaluator to detect hallucinations in model outputs.

**Ensure `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` are correctly set in your `.env` file.**

**Evaluator Prompt**
```
You are an expert data labeler evaluating model outputs for hallucinations. Your task is to determine whether the output contains hallucinations, based solely on your own knowledge, and provide a brief explanation.

<Rubric>
  A response without hallucinations:
  - Contains only verifiable facts
  - Makes no unsupported claims or assumptions
  - Does not add speculative or imagined details
  - Maintains accuracy in dates, numbers, and specific details
  - Appropriately indicates uncertainty when information is incomplete
</Rubric>

<Instructions>
  - Read the output thoroughly
  - Identify all claims made in the output
  - Determine whether each claim is supported by general factual knowledge
  - Note any claims that are likely hallucinations (unsupported, false, or fabricated)
  - Consider the severity and quantity of hallucinations
</Instructions>

<Reminder>
  Focus solely on factual accuracy based on your knowledge. Do not rely on any external input context, and do not evaluate style, grammar, or presentation. A shorter, factually correct response is preferred over a longer response with unsupported claims.
</Reminder>

Question:
{{inputs}}

<output>
{{outputs}}
</output>

```

### Evaluation Metrics

**Hallucination Score** (0-1): Measures if answer is grounded in PDF sources
   - 1.0 = Hallucinated
   - 0.0 = No hallucinations, fully grounded
  
## Usage

### Run the Streamlit App

```bash
streamlit run main.py
```

Then:
1. Enter your question about statistical learning
2. Click "Generate"
3. View the answer with source citations
4. Download the markdown file with complete references

## LangSmith Monitoring

All RAG operations are automatically traced to LangSmith:

1. **View traces**: https://smith.langchain.com/
2. **Project name**: `RAG_assistant_evaluator` (as set in `.env`)

### Track Information:

- LLM calls & token usage
- Vector search results
- Graph node transitions
- Full input/output traces
- Evaluation scores

## Output Format

### Example Output

```markdown
### Answer:
Linear regression is a statistical method for modeling the relationship
between a dependent variable and one or more independent variables [Source 1].
The method uses least squares estimation to find the best-fitting line [Source 2].

---

## Sources:

**Source 1** (Page 45):
Excerpt: Linear regression models the relationship between variables by
fitting a linear equation to observed data...

**Source 2** (Page 47):
Excerpt: The least squares method minimizes the sum of squared residuals
between observed and predicted values...
```

### Downloaded Markdown

Each response is saved with timestamp:

```
response_20250128_143022.md
```

Format:
```markdown
# Question
What is linear regression?

## Answer
[Answer with citations]

## Sources:
[Source references with page numbers and excerpts]
```

## Configuration

### State Schema (graph.py)

```python
class RAGState(TypedDict):
    query: str      # User question
    answer: str     # LLM-generated answer
    source: str     # Formatted source references
```

### Vector Store Settings

- **Chunk size**: 700 
- **Chunk overlap**: 100 
- **Similarity search**: 4 
- **Embedding**: OpenAI 
- **Vector DB**: FAISS 

### LLM Configuration

- **Model**: GPT-4o
- **Temperature**: 0.1 
- **Context**: 4 retrieved chunks

## Project Structure

### graph.py
Main RAG workflow using LangGraph:
- PDF loading and chunking
- Vector store creation
- Query processing node
- Source extraction and formatting

### main.py
Streamlit interface:
- User input form
- Response display
- Source visualization
- Markdown export

### config.py
Configuration management:
- Load API keys from `.env`
- Environment variable setup

### Monitor and Evaluate on LangSmith 
LLM-as-a-Judge evaluation framework:
- Hallucination detection


## Performance

- **Vector store initialization**: ~50-60 seconds (first run only)
- **Query processing**: ~2-5 seconds per query
- **Token usage**: ~1,000-2000 tokens per query (depending on context)


## LangSmith Dashboard

View comprehensive monitoring at https://smith.langchain.com/

**Available Insights:**
- Request traces with timing
- Token usage per query
- Error tracking
- Input/output inspection
- Evaluation (need to create evaluators on your own)

## License

MIT License

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Platform](https://smith.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
