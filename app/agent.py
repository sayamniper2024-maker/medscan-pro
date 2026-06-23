"""
agent.py

Core agent logic: talking to Groq (the LLM) and Tavily (web search),
wired together into a tool-calling research loop.

This is a direct port of the Colab notebook logic - no new behavior,
just reorganized into a proper module so it can be imported by the
FastAPI app instead of living in notebook cells.
"""

import os
import json
import time
import random
import datetime

from groq import Groq
from tavily import TavilyClient


# ---------------------------------------------------------------------------
# Configuration & API clients
# ---------------------------------------------------------------------------

# In Colab, these came from userdata.get(...). On a real server, the
# equivalent is environment variables - same idea, different "panel".
GROQ_API_KEY = os.environ.get("GROQ_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_KEY environment variable is not set.")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

MODEL_NAME = "openai/gpt-oss-120b"

# Toggle this to True during development to avoid spending real tokens.
# (Matches the MOCK_MODE switch from the Colab notebook.)
MOCK_MODE = os.environ.get("MEDSCAN_MOCK_MODE", "false").lower() == "true"

# Tool definition the model can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current, real-world information. Use this "
                "whenever you need facts, figures, company names, market data, "
                "or anything that could be outdated in your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_SYSTEM_PROMPT = (
    "You are a research assistant with a STRICT search budget of 3 searches maximum per question. "
    "Track your search count internally. On your first search, cast a broad net. On your second "
    "search, fill in the most important gap. On your third search, fill any remaining gap if "
    "needed - then you MUST stop searching and write your final answer using whatever "
    "information you have gathered, even if incomplete. "
    "Never search for the same fact twice. If a precise number isn't in your search results, "
    "report the closest approximate figure you found, name its source, and move on. "
    "Precision-chasing is not worth additional searches - a well-sourced approximate answer "
    "is the correct outcome, not a failure."
)


# ---------------------------------------------------------------------------
# Daily token budget tracking (persists across restarts via a local file)
# ---------------------------------------------------------------------------

DAILY_TOKEN_LIMIT = 200_000
SAFETY_BUFFER = 10_000
USAGE_FILE = "/tmp/medscan_daily_usage.json"

daily_tokens_used = 0


def load_daily_usage():
    today = str(datetime.date.today())
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        if data.get("date") == today:
            return data.get("tokens_used", 0)
    return 0


def save_daily_usage(tokens_used):
    today = str(datetime.date.today())
    with open(USAGE_FILE, "w") as f:
        json.dump({"date": today, "tokens_used": tokens_used}, f)


daily_tokens_used = load_daily_usage()


def record_usage(tokens_used):
    global daily_tokens_used
    daily_tokens_used += tokens_used
    save_daily_usage(daily_tokens_used)


def check_daily_budget():
    remaining = DAILY_TOKEN_LIMIT - daily_tokens_used
    return remaining > SAFETY_BUFFER


def print_budget_status():
    remaining = DAILY_TOKEN_LIMIT - daily_tokens_used
    print(
        f"Daily tokens used so far: {daily_tokens_used:,} / {DAILY_TOKEN_LIMIT:,} "
        f"({remaining:,} remaining)"
    )


# ---------------------------------------------------------------------------
# Search archive (full, untruncated text - used later for fabrication checks)
# ---------------------------------------------------------------------------

full_search_archive = []


# ---------------------------------------------------------------------------
# Mock mode helpers (free, instant, no real API calls - for pipeline testing)
# ---------------------------------------------------------------------------

def mock_search_web(query, max_chars=500, max_results=2):
    fake_result = (
        f"Title: Mock result for '{query}'\n"
        f"Content: This is simulated search content for testing purposes. "
        f"In a real run, this would contain actual web search results about {query}. "
        f"Example company mentioned: TestCorp Surgical. Example figure: $5 billion market (2024, Mock Source).\n\n"
    )
    full_search_archive.append(fake_result)
    return fake_result


def mock_groq_response(messages, use_tools=True):
    class FakeUsage:
        total_tokens = random.randint(400, 1200)

    class FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class FakeToolCall:
        def __init__(self, query):
            self.id = f"mock_call_{random.randint(1000, 9999)}"

            class FakeFunction:
                name = "search_web"
                arguments = json.dumps({"query": query})

            self.function = FakeFunction()

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, message):
            self.choices = [FakeChoice(message)]
            self.usage = FakeUsage()

    def is_tool_message(m):
        if isinstance(m, dict):
            return m.get("role") == "tool"
        return getattr(m, "role", None) == "tool"

    num_previous_tool_results = sum(1 for m in messages if is_tool_message(m))

    if use_tools and num_previous_tool_results < 1:
        fake_call = FakeToolCall("mock search query")
        return FakeResponse(FakeMessage(content=None, tool_calls=[fake_call]))
    else:
        fake_answer = (
            "**Mock Final Answer**\n\nBased on simulated search results, the mock market size "
            "is approximately $5 billion (Mock Source, 2024). This is placeholder content "
            "generated in MOCK_MODE for testing the pipeline without using real API tokens."
        )
        return FakeResponse(FakeMessage(content=fake_answer, tool_calls=None))


# ---------------------------------------------------------------------------
# Real search function (routes to mock when MOCK_MODE is on)
# ---------------------------------------------------------------------------

def search_web(query, max_chars=500, max_results=2):
    if MOCK_MODE:
        return mock_search_web(query, max_chars, max_results)

    results = tavily_client.search(query=query, max_results=max_results)
    formatted = ""
    for r in results["results"]:
        full_content = r["content"]
        full_search_archive.append(full_content)
        truncated_content = full_content[:max_chars]
        formatted += f"Title: {r['title']}\nContent: {truncated_content}\n\n"
    return formatted


# ---------------------------------------------------------------------------
# The core agent loop
# ---------------------------------------------------------------------------

def run_agent(user_message, system_prompt=None, max_searches=3, debug=True):
    """
    Runs a tool-calling agent loop: the model can call search_web up to
    max_searches times, then is forced to produce a final synthesized answer.

    Returns: (answer_text, collected_search_text_list)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    search_count = 0
    collected_search_text = []

    while search_count < max_searches:
        if not MOCK_MODE and not check_daily_budget():
            if debug:
                print("Daily token budget nearly exhausted - stopping early.")
            return "Stopped early: daily token budget exhausted for today.", collected_search_text

        try:
            if MOCK_MODE:
                response = mock_groq_response(messages, use_tools=True)
            else:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
        except Exception as e:
            if debug:
                print("Error during search phase:", e)
            break

        tokens_used = response.usage.total_tokens
        if not MOCK_MODE:
            record_usage(tokens_used)

        if debug:
            print(f"Tokens used this call: {tokens_used}{'  [MOCK]' if MOCK_MODE else ''}")
            if not MOCK_MODE:
                print_budget_status()
        if not MOCK_MODE and tokens_used > 3000:
            if debug:
                print("High token usage detected, pausing 20s as a precaution...")
            time.sleep(20)

        response_message = response.choices[0].message
        messages.append(response_message)

        if debug:
            print(f"--- Search {search_count + 1} ---")
            print("Content:", response_message.content)
            print("Tool calls:", response_message.tool_calls)
            print()

        if not response_message.tool_calls:
            return response_message.content, collected_search_text

        for tool_call in response_message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", "")
                if debug:
                    print("Searching for:", query)
                result = search_web(query) if query else "Invalid search query, no results."
            except Exception:
                result = "Invalid search request, no results."

            collected_search_text.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        search_count += 1

    summary_prompt = (
        "Based on all the search results above, write your final answer now. "
        "Do not attempt to search again. Summarize what you found, note any "
        "figures that are approximate, and cite sources where possible."
    )
    messages.append({"role": "user", "content": summary_prompt})

    try:
        if MOCK_MODE:
            final_response = mock_groq_response(messages, use_tools=False)
        else:
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
        if not MOCK_MODE:
            record_usage(final_response.usage.total_tokens)
        return final_response.choices[0].message.content, collected_search_text
    except Exception as e:
        return f"Final answer generation also failed: {e}", collected_search_text
