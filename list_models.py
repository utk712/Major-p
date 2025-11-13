from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai, os
os.environ.setdefault('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY'))
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
print('configured')
try:
    models = genai.list_models()
    print('Listing up to 50 models:')
    count=0
    for m in models:
        nm = getattr(m,'name',None)
        print('-', nm)
        count+=1
        if count>50:
            break
except Exception as e:
    print('error listing models:', repr(e))