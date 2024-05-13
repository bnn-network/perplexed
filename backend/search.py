from enum import Enum
import bs4
import concurrent.futures
import datetime
import groq
import html
import os
import pprint
import requests
import json
import sys
import time
import re
from typing import List, Callable
import urllib.parse
from typing import Callable, List
import sys

from query_cache import QueryCache

current_dir_path = os.path.dirname(os.path.realpath(__file__))
with open(current_dir_path + '/config.json') as config_file:
    CONFIG = json.load(config_file)
DOMAINS_ALLOW = CONFIG['DOMAINS_ALLOW']
JSON_STREAM_SEPARATOR = "[/PERPLEXED-SEPARATOR]"
GROQ_CLIENT = groq.Groq(api_key=CONFIG['GROQ_API_KEY'])
GROQ_MODEL = 'llama3-8b-8192'
GROQ_LIMIT_TOKENS_PER_MINUTE = 30000
WEBSEARCH_DOMAINS_BLACKLIST = ["quora.com", "www.quora.com"]
WEBSEARCH_RESULT_MIN_TOKENS = 50
WEBSEARCH_NUM_RESULTS_SLICE = 4
WEBSEARCH_READ_TIMEOUT_SECS = 5
WEBSEARCH_CONNECT_TIMEOUT_SECS = 3
WEBSEARCH_CONTENT_LIMIT_TOKENS = 1000

query_cache = QueryCache()

class WebSearchDocument:
    def __init__(self, id, title, url, text=''):
        self.id = id
        self.title = html.escape(title)
        self.url = url
        self.text = html.escape(text)
    
    def __str__(self) -> str:
        return f"{self.title}\n{self.url}\n{self.text[:100]}"

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'text': self.text
        }

def print_log(*args, **kwargs):
    datestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('[' + datestr + ']', *args, file=sys.stderr, **kwargs)

def limit_tokens(input_string: str, N: int) -> str:
    tokens = input_string.split()
    limited_tokens = tokens[:N]
    return ' '.join(limited_tokens)

def count_tokens(input_string: str) -> int:
    tokens = input_string.split()
    return len(tokens)

def query_websearch(query: str, original_topic: str) -> list[WebSearchDocument]:
    if not query.strip():
        return []  # Return an empty list if the search query is empty or whitespace

    url = f"https://www.googleapis.com/customsearch/v1?key={CONFIG['GOOGLE_SEARCH_API_KEY']}&cx={CONFIG['GOOGLE_SEARCH_ENGINE_ID']}&q={query}"
    response = requests.get(url)
    blob = response.json()
    if 'items' not in blob:
        print(f"Error querying Google: {blob}")
        return []
    results = blob['items']
    ret: list[WebSearchDocument] = []
    id = 0
    for result in results:
        link = result['link']
        title = result['title']
        snippet = result['snippet']
        link_parsed = urllib.parse.urlparse(link)
        if link_parsed.netloc in WEBSEARCH_DOMAINS_BLACKLIST:
            continue
        id += 1
        if id > WEBSEARCH_NUM_RESULTS_SLICE:
            break
        if original_topic.lower() in title.lower() or original_topic.lower() in snippet.lower():
            # Prioritize results that are relevant to the original topic
            ret.insert(0, WebSearchDocument(id=id, title=title, url=link))
        else:
            ret.append(WebSearchDocument(id=id, title=title, url=link))
    return ret

def scrape_webpage(url: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    }
    response = requests.get(url, timeout=(WEBSEARCH_CONNECT_TIMEOUT_SECS, WEBSEARCH_READ_TIMEOUT_SECS), headers=headers)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')

    # Extract main text from the webpage
    # This will vary depending on the structure of the webpage
    # Here we are assuming that the main text is within <p> tags
    main_text = ' '.join([p.text for p in soup.find_all('p')])
    main_text = limit_tokens(main_text, WEBSEARCH_CONTENT_LIMIT_TOKENS)

    return main_text

def scrape_webpage_threaded(websearch_doc):
    try:
        text = scrape_webpage(websearch_doc.url)
    except Exception as e:
        print(f"Error scraping {websearch_doc.url}: {e}")
        text = ""
    websearch_doc.text = text
    return websearch_doc

def replace_documents_with_markdown(text, websearch_docs):
    def replace_doc_id(match):
        doc_id = match.group(1)
        doc = next((doc for doc in websearch_docs if doc.id == int(doc_id)), None)
        if doc:
            return f"[{doc_id}]({doc.url})"
        else:
            return f"[{doc_id}]"

    return re.sub(r'\[(\d+)\]', replace_doc_id, text)

def generate_search_query(user_prompt, conversation_history):
    messages = [
        {
            "role": "system",
            "content": "You are a search query generator. Given a conversation history and a user prompt, generate a single relevant search query to find information related to the user's question. Keep the query simple and general."
        },
        {
            "role": "user",
            "content": f"Conversation History:\n{conversation_history}\n\nUser Prompt:\n{user_prompt}\n\nGenerate a search query based on the conversation history and the user prompt."
        }
    ]

    response = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=[],
        max_tokens=50
    )

    search_query = response.choices[0].message.content.strip().strip('"')
    print_log(f"Generated search query: {search_query}")  # Log the generated search query

    return search_query

def query_chatbot(user_prompt, websearch_docs, conversation_history):
    value = query_cache.get(user_prompt)
    if value:
        print_log("query_chatbot: cache hit for:", user_prompt)
        return value

    if user_prompt.lower() in ["who are you", "who made you"]:
        response_message = "I am ePiphany, an AI assistant created, owned, and operated by ePiphany AI."
    else:
        if websearch_docs:
            content_docs = prepare_content_docs(websearch_docs)
            system_prompt = get_system_prompt()
            
            # Include the conversation history in the context
            conversation_history_str = '\n'.join([f"User: {entry['userPrompt']}\nAssistant: {entry['assistantResponse']}" for entry in conversation_history])
            
            # Generate a search query based on the user prompt and conversation history
            search_query = generate_search_query(user_prompt, conversation_history_str)
            
            system_content = prepare_system_content(system_prompt, content_docs, conversation_history_str, search_query)
            messages = prepare_messages(system_content, search_query)
            response_message = get_chatbot_response(messages, websearch_docs)
        else:
            response_message = "I apologize, but I couldn't find any relevant information to answer your question."

    if user_prompt and response_message:
        query_cache.set(user_prompt, response_message)

    return response_message

def prepare_content_docs(websearch_docs):
    content_docs = ""
    for doc in websearch_docs:
        num_tokens = count_tokens(doc.text)
        if num_tokens < WEBSEARCH_RESULT_MIN_TOKENS:
            continue
        content_docs += f"====\nDOCUMENT ID:{doc.id}\nDOCUMENT TITLE:{doc.title}\nDOCUMENT URL:{doc.url}\nDOCUMENT TEXT:{doc.text}\n"
    return content_docs

def get_system_prompt():
    return """
    You are an AI assistant named ePiphany, created by ePiphany AI, that helps users find answers to their questions. When a user asks a question, search for relevant information from the provided documents and generate a comprehensive response that directly addresses their query. Integrate insights from multiple sources to provide a well-informed answer.

    Respond to the user's question as if you are having a friendly conversation with them. Use a warm and engaging tone, and address the user directly using "you" and "your." Avoid sounding like an article or a formal report.

    Format your response using the following guidelines:

    - Use double newline characters (\\n\\n) to separate paragraphs.
    - Use bullet points (- ) to present information in a structured manner, if necessary.
    - Use bold (**text**) to emphasize important points, if necessary.
    - Use italics (*text*) to add variety to your response, if necessary.
    - Use citation markers in square brackets ([number]) to indicate the source of information, where [number] corresponds to the document ID.

    Format the answer as markdown. After each sentence, cite the document information used using the exact syntax "[<ID>]". Check over your work. Remember to make your work clear and concise. Remember to cite the source after each sentence with the syntax "[ID]".

    When citing information from the provided documents, use citation markers in square brackets, such as [1], [2], etc. Place these markers immediately after the relevant text.
    
    Aim to create a friendly, engaging, and informative tone in your response. Provide a comprehensive answer that satisfies the user's curiosity and leaves them feeling well-informed. Speak directly to the user as if you are ePiphany AI, without mentioning the names of the sources.

    Remember to make your response clear, concise, and well-structured. Cite the relevant documents using citation markers in square brackets.
    """

def prepare_system_content(system_prompt, content_docs, conversation_history, user_prompt):
    return f"====SYSTEM PROMPT:{system_prompt}\n{content_docs}\n====CONVERSATION HISTORY:\n{conversation_history}\n====QUESTION: {user_prompt}"

def prepare_messages(system_content, user_prompt):
    return [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

def get_chatbot_response(messages, combined_websearch_docs):
    content_docs = prepare_content_docs(combined_websearch_docs)
    system_content = messages[0]["content"] + "\n" + content_docs

    # Truncate the system content if it exceeds the allowed length
    max_system_content_length = 8000  # Adjust this value based on the allowed context length
    if len(system_content) > max_system_content_length:
        system_content = system_content[:max_system_content_length]

    response = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": messages[1]["content"]
            }
        ],
        tools=[],
        max_tokens=4096
    )
    return response.choices[0].message.content

def post_process_response(response_message, user_prompt, websearch_docs):
    response_message = response_message.replace(user_prompt, "").strip()
    response_message = replace_documents_with_markdown(response_message, websearch_docs)
    return response_message

def extract_original_topic(conversation_history):
    if conversation_history:
        first_user_prompt = conversation_history.split("User: ")[1].split("\n")[0]
        return first_user_prompt
    return ""

class SearchAllStage(Enum):
    STARTING = "Starting search"
    QUERIED_GOOGLE = "Querying Google"
    DOWNLOADED_WEBPAGES = "Downloading Webpages"
    QUERIED_LLM = "Querying LLM"
    RESULTS_READY = "Results ready"