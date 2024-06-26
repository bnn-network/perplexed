from typing import List
import concurrent.futures
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import html
import json
from search import SearchAllStage, WebSearchDocument, count_tokens, print_log, query_chatbot, query_websearch, scrape_webpage_threaded, DOMAINS_ALLOW, GROQ_LIMIT_TOKENS_PER_MINUTE, JSON_STREAM_SEPARATOR, extract_original_topic, generate_search_query
import time

app = Flask(__name__)
CORS(app, resources={r"/stream_search": {"origins": DOMAINS_ALLOW}})


@app.route('/test', methods=['GET'])
def test():
    print_log("test ping")
    return 'HELLO'

class StreamSearchResponse:
    def __init__(self, success: bool, stage: SearchAllStage, num_tokens_used: int, websearch_docs: List[WebSearchDocument], answer="") -> None:
        self.success = success
        self.stage = stage
        self.num_tokens_used = num_tokens_used
        self.websearch_docs = websearch_docs
        self.answer = answer

    def to_json_data(self):
        return (json.dumps({
                    'success': self.success,
                    'stage': self.stage.value,
                    'num_tokens_used': self.num_tokens_used,
                    'websearch_docs': [doc.to_dict() for doc in self.websearch_docs],
                    'answer': self.answer
        }) + JSON_STREAM_SEPARATOR).encode('utf-8') 

@app.route('/stream_search', methods=['POST'])
def stream_search():
    data = request.get_json()
    user_prompt = data.get('user_prompt')
    conversation_history = data.get('conversation_history', '')

    print_log("stream_search query:", user_prompt)
    if not user_prompt:
        return jsonify({'success': False, 'message': 'Please provide a user prompt.'})
    def generate(conversation_history):
        original_topic = extract_original_topic(conversation_history)
        websearch_docs: list[WebSearchDocument] = query_websearch(user_prompt, original_topic)

        yield StreamSearchResponse(success=True, stage=SearchAllStage.QUERIED_GOOGLE, num_tokens_used=0, websearch_docs=websearch_docs).to_json_data()

        websearch_docs_scraped = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            websearch_docs_scraped = list(executor.map(scrape_webpage_threaded, websearch_docs))

        num_tokens_used = sum([count_tokens(doc.text) for doc in websearch_docs_scraped])

        yield StreamSearchResponse(success=True, stage=SearchAllStage.DOWNLOADED_WEBPAGES, num_tokens_used=num_tokens_used, websearch_docs=websearch_docs_scraped).to_json_data()

        conversation_history_str = '\n'.join([f"User: {entry['userPrompt']}\nAssistant: {entry['assistantResponse']}" for entry in conversation_history])
        answer = query_chatbot(user_prompt, websearch_docs_scraped, conversation_history_str)

        # Limit conversation history to 4000 tokens
        conversation_history_tokens = conversation_history_str.split()
        conversation_history_tokens = conversation_history_tokens[-4000:]
        conversation_history_str = ' '.join(conversation_history_tokens)

        yield StreamSearchResponse(success=True, stage=SearchAllStage.RESULTS_READY, num_tokens_used=num_tokens_used, websearch_docs=websearch_docs_scraped, answer=answer).to_json_data()

        return conversation_history

    updated_conversation_history = generate(conversation_history)
    return Response(stream_with_context(updated_conversation_history), mimetype='application/json')

if __name__ == "__main__":
    # app.run(debug=False, host='0.0.0.0', port=80)
    app.run(debug=False)