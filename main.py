#Import Libraries
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from newspaper import Article
import random
import string
import nltk
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings


#Initialize FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#Ignore Warnings
warnings.filterwarnings('ignore')


#Download NLTK Packages
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize chatbot components
# Initialize chatbot components
class BilingualChatbot:
    def __init__(self):
        # Common initialization
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        
        # English initialization
        self.initialize_english()
        
        # Arabic initialization
        self.initialize_arabic()
    
    def initialize_english(self):
        # English article setup
        english_article = Article('https://en.wikipedia.org/wiki/Constitution_of_Lebanon')
        english_article.download()
        english_article.parse()
        english_article.nlp()
        self.english_corpus = english_article.text
        self.sent_tokensEN = nltk.sent_tokenize(self.english_corpus)
        
        # English greetings
        self.english_greeting_input = ["hi", "hello", "hey", "hola"]
        self.english_greeting_response = ["howdy", "hey there", "hi", "hello :)"]
    
    def initialize_arabic(self):
        # Arabic article setup
        arabic_article = Article('https://ar.wikipedia.org/wiki/دستور_لبنان')
        arabic_article.language = 'ar'
        arabic_article.download()
        arabic_article.parse()
        arabic_article.nlp()
        self.arabic_corpus = arabic_article.text
        self.sent_tokensAR = nltk.sent_tokenize(self.arabic_corpus)
        
        # Arabic greetings
        self.arabic_greeting_input = ["مرحبا", "أهلا"]
        self.arabic_greeting_response = ["مرحبا بكم"]
        
        # Arabic stopwords
        self.arabic_stopwords = [
            'في', 'من', 'عن', 'إلى', 'على', 'هذا', 'هذه', 'هم', 'هناك', 'هي', 'هو',
            'أنت', 'أنا', 'نحن', 'إذا', 'عند', 'على', 'عليه', 'عليها', 'عليهم',
            'عليهن', 'علينا', 'أنا', 'أنت', 'أنتم', 'أنتن', 'إياك', 'إياه', 'إياها',
            'إياهم', 'إياهن', 'إياهما', 'إيانا', 'أنتما', 'إيانك', 'أنتن', 'هي', 'هو'
        ]
    
    def lem_normalize(self, text):
        return nltk.word_tokenize(text.lower().translate(self.remove_punct_dict))
    
    def arabic_tokenizer(self, text):
        return nltk.word_tokenize(text)
    
    def english_greeting(self, sentence):
        for word in sentence.split():
            if word.lower() in self.english_greeting_input:
                return random.choice(self.english_greeting_response)
        return None
    
    def arabic_greeting(self, sentence):
        for word in sentence.split():
            if word.lower() in self.arabic_greeting_input:
                return random.choice(self.arabic_greeting_response)
        return None
    
    def english_response(self, user_response):
        user_response = user_response.lower()
        robo_response = ''
        self.sent_tokensEN.append(user_response)
        
        tfidfvec = TfidfVectorizer(tokenizer=self.lem_normalize, stop_words='english')
        tfidf = tfidfvec.fit_transform(self.sent_tokensEN)
        val = cosine_similarity(tfidf[-1], tfidf)
        idx = val.argsort()[0][-2]
        flat = val.flatten()
        flat.sort()
        score = flat[-2]
        
        if score == 0:
            robo_response = robo_response + "Sorry, I don't understand"
        else:
            robo_response = robo_response + self.sent_tokensEN[idx]
        
        self.sent_tokensEN.remove(user_response)
        return robo_response
    
    def arabic_response(self, user_response):
        user_response = user_response.lower()
        robo_response = ''
        self.sent_tokensAR.append(user_response)
        
        tfidfvec = TfidfVectorizer(tokenizer=self.arabic_tokenizer, stop_words=self.arabic_stopwords)
        tfidf = tfidfvec.fit_transform(self.sent_tokensAR)
        val = cosine_similarity(tfidf[-1], tfidf)
        idx = val.argsort()[0][-2]
        flat = val.flatten()
        flat.sort()
        score = flat[-2]
        
        if score == 0:
            robo_response = robo_response + "أعتذر ليس لدي معلومات حول هذا الموضوع"
        else:
            robo_response = robo_response + self.sent_tokensAR[idx]
        
        self.sent_tokensAR.remove(user_response)
        return robo_response

# Initialize chatbot
chatbot = BilingualChatbot()



# API endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(language: str = Form(...), message: str = Form(...)):
    if language == '1':  # English
        if message.lower() in ['thanks', 'thank you', 'bye']:
            return {"response": "You're welcome! Type a new message to continue chatting."}
        
        greeting = chatbot.english_greeting(message)
        if greeting is not None:
            return {"response": greeting}
        else:
            return {"response": chatbot.english_response(message)}
    
    elif language == '2':  # Arabic
        if message.lower() in ['شكرا', 'شكرا لك', 'مع السلامة']:
            return {"response": "على الرحب والسعة! اكتب رسالة جديدة لمواصلة الدردشة."}
        
        greeting = chatbot.arabic_greeting(message)
        if greeting is not None:
            return {"response": greeting}
        else:
            return {"response": chatbot.arabic_response(message)}
    
    else:
        return {"response": "Invalid language selection"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)