"""Small OpenAI-compatible transport implementing MedAgentSim's BAgent API."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

class BAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = os.environ.get('EGIP_API_BASE', 'http://127.0.0.1:8000/v1').rstrip('/')
        self.timeout = float(os.environ.get('EGIP_TIMEOUT', '120'))
        self.retries = int(os.environ.get('EGIP_RETRIES', '2'))
        self.temperature = float(os.environ.get('EGIP_TEMPERATURE', '0.2'))
        self.max_tokens = int(os.environ.get('EGIP_MAX_TOKENS', '200'))
        self.disable_thinking = os.environ.get('EGIP_DISABLE_THINKING', '1') == '1'
        if self.retries < 1 or self.max_tokens < 1 or self.timeout <= 0:
            raise ValueError('Retries, max tokens and timeout must be positive')
        info = self._request('/models')
        available = [entry['id'] for entry in info.get('data', [])]
        if model_name not in available:
            raise ValueError(f'Model {model_name!r} is not served. Available models: {available}')

    def _request(self, path, payload=None):
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get('EGIP_API_KEY')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        body = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request(self.base_url + path, data=body, headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def query_model(self, prompt, system_prompt='You are a helpful assistant.',
                    tries=None, timeout=None, image_requested=False, **kwargs):
        if image_requested:
            raise ValueError('The reference experiment uses text inputs only')
        # `timeout` in the native scorer controls its retry delay. HTTP timeouts
        # are configured separately through EGIP_TIMEOUT.
        payload = {'model': self.model_name,
                   'messages': [{'role': 'system', 'content': str(system_prompt)},
                                {'role': 'user', 'content': str(prompt)}],
                   'max_tokens': self.max_tokens, 'temperature': self.temperature}
        if self.disable_thinking:
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        attempts = self.retries if tries is None else tries
        if attempts < 1:
            raise ValueError('tries must be positive')
        for attempt in range(attempts):
            try:
                response = self._request('/chat/completions', payload)
                text = response['choices'][0]['message']['content']
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError('Model returned an empty text response')
                return text.strip()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                # Configuration errors must not silently fall back to another model.
                if isinstance(exc, HTTPError) and exc.code < 500 and exc.code != 429:
                    raise RuntimeError(f'Model request failed: HTTP {exc.code}') from exc
                if attempt + 1 == attempts:
                    raise RuntimeError('Model service unavailable after retries') from exc
                time.sleep(1)
