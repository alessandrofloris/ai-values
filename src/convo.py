import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Some models (e.g. Gemma) prefix the answer with a reasoning block like
# <thought>...</thought>5. Strip such blocks so only the actual reply is scored.
_THINK_RE = re.compile(r'<(thought|thinking|think)>.*?</\1>', re.IGNORECASE | re.DOTALL)
# Others (e.g. Gemini 4 Argon) leak a bare channel label before the answer:
# 'thought\n2', 'sthought\n6', 'thinking\n4', or the Chinese '思想7' ('thought'). Only a
# short label on its own line (or a CJK label) is removed, so real replies starting
# "Thinking about..." survive.
_THINK_LABEL_RE = re.compile(r'^\s*(?:\w{0,3}(?:thoughts?|thinking|think)[ \t]*:?[ \t]*\n|(?:思想|思考|思维)[:：]?)',
                             re.IGNORECASE)


def _strip_thoughts(text: str) -> str:
    return _THINK_LABEL_RE.sub('', _THINK_RE.sub('', text).strip()).strip()


def _text_of(content) -> str:
    """Some reasoning models (e.g. Mistral Large 4) return content as a list of chunks,
    [{'type': 'thinking', ...}, {'type': 'text', 'text': ...}]. Keep only the text chunks."""
    if isinstance(content, list):
        return ''.join(c.get('text', '') for c in content
                       if isinstance(c, dict) and c.get('type') == 'text')
    return content

PROVIDER_URLS = {
    'foundry': 'https://mtha-testbed-proj-resource.openai.azure.com/openai/v1',
    'anthropic': 'https://api.anthropic.com/v1',
    'deepseek': 'https://api.deepseek.com',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai',
    'huggingface': 'https://router.huggingface.co/v1',
    'mistral': 'https://api.mistral.ai/v1',
    'openai': 'https://api.openai.com/v1',
    'perplexity': 'https://api.perplexity.ai',
    'qwen': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    'xai': 'https://api.x.ai/v1',
    'meta': 'https://api.meta.ai/v1',
}

CUSTOM_PROVIDER_KEYS = {'custom_openai', 'custom', 'custom-openai', 'local'}


class Convo:
    """A stateful conversation with a single LLM, using the OpenAI-compatible chat interface."""

    def __init__(self, provider: str, model: str = None, system: str = None):
        # Provider strings keep their display capitalisation in config/experiments.json
        # (e.g. "Gemini", "OpenAI", "custom_openai"); casefold for lookup.
        key = provider.casefold()

        if key in CUSTOM_PROVIDER_KEYS:
            base_url = os.environ.get('CUSTOM_OPENAI_BASE_URL')
            if not base_url:
                raise ValueError(
                    "Provider 'custom_openai' requires CUSTOM_OPENAI_BASE_URL to be set in .env"
                )
            api_key = os.environ.get('CUSTOM_OPENAI_API_KEY') or 'dummy-key'

            if not model or model.casefold() == 'default':
                model = os.environ.get('CUSTOM_OPENAI_MODEL', 'default')

            timeout = float(os.environ.get('CUSTOM_OPENAI_REQUEST_TIMEOUT', 120))
            max_retries = int(os.environ.get('CUSTOM_OPENAI_MAX_RETRIES', 5))
        else:
            if key not in PROVIDER_URLS:
                raise ValueError(f'Unknown provider {provider!r}. Known: {list(PROVIDER_URLS) + ["custom_openai"]}')
            base_url = PROVIDER_URLS[key]
            api_key = os.environ[f'{key.upper()}_API_KEY']
            timeout = None
            max_retries = None

        client_kwargs = {
            'base_url': base_url,
            'api_key': api_key,
        }
        if timeout is not None:
            client_kwargs['timeout'] = timeout
        if max_retries is not None:
            client_kwargs['max_retries'] = max_retries

        self.provider = provider
        self.model = model
        self.messages = [{'role': 'system', 'content': system or 'You are a helpful assistant.'}]
        self.log_data = []
        self._client = OpenAI(**client_kwargs)

    def ask(self, prompt: str, reasoning_effort: str = None) -> str:
        """Send a message and return the reply. Appends both to conversation history."""
        self.messages.append({'role': 'user', 'content': prompt})
        params = {'model': self.model, 'messages': self.messages}
        if reasoning_effort is not None:
            params['reasoning_effort'] = reasoning_effort
        response = self._client.chat.completions.create(**params)
        content = ''
        if response.choices:
            msg = response.choices[0].message
            raw_content = msg.content
            # Fallback for reasoning models (e.g. DeepSeek-R1 / Ollama / vLLM) where content is null
            if not raw_content:
                raw_content = getattr(msg, 'reasoning_content', None)
                if not raw_content and hasattr(msg, 'model_extra') and isinstance(msg.model_extra, dict):
                    raw_content = msg.model_extra.get('reasoning_content')
            content = _text_of(raw_content) or ''
            if content:
                content = _strip_thoughts(content)

        self.log_data.append(response)
        self.messages.append({'role': 'assistant', 'content': content})
        return content

    def history(self) -> list:
        return self.messages

    def log(self) -> list:
        return self.log_data
