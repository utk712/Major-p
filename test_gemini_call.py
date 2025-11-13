from dotenv import load_dotenv
load_dotenv()
import os
import app
print('ensure=', app.ensure_gemini_configured())
try:
    resp = app.gemini_chat_response([
        {'author':'system','content':'You are a helpful assistant.'},
        {'author':'user','content':'Hello, say hi.'}
    ])
    print('response:', resp)
except Exception as e:
    print('error calling gemini:', repr(e))