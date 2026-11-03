"""Local HTTP fixture for automated integration tests."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading

@contextmanager
def mock_server():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_json(self, value, status=200):
            body=json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.send_json({'data':[{'id':'mock'}]})
        def do_POST(self):
            req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(req)
            system,prompt=[x['content'] for x in req['messages']]
            if 'medical moderator' in system:
                text='[0.00]'  # Arbitrary mock score, unrelated to clinical correctness.
            elif 'diagnostic doctor' in system:
                text='DIAGNOSIS READY: MOCK ANSWER'
            elif system.startswith('You are an measurement reader'):
                text='RESULTS: MOCK OBSERVATION'
            elif 'role: patient' in system:
                text='MOCK PATIENT RESPONSE'
            else:
                menu=re.findall(r'A_[a-f0-9]{12}',system)
                text='REQUEST TEST: '+menu[0] if menu else 'When did it begin?'
            self.send_json({'choices':[{'message':{'content':text}}]})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1',calls
    finally:
        server.shutdown();server.server_close();thread.join()
