import importlib, traceback, os

# Import the app module (does not start the server because __name__ != '__main__')
app = importlib.import_module('app')

try:
    print('Calling gemini_chat...')
    res = app.gemini_chat([{'author': 'user', 'content': 'hello from test'}])
    print('RESPONSE:', res)
except Exception:
    traceback.print_exc()
    # Also show some helpful state
    print('\nGEMINI_API_KEY set:', bool(os.getenv('GEMINI_API_KEY')))
    try:
        import google.generativeai as genai
        print('genai module:', genai)
        print('genai has ChatSession:', hasattr(genai, 'ChatSession'))
    except Exception as e:
        print('Failed to import genai for debug:', e)