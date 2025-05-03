from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os
import json
import datetime
import re
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from bson import ObjectId

# Load environment variables
load_dotenv()

# MongoDB connection string
MONGODB_URL = os.getenv("MONGODB_URL",
                        "mongodb+srv://username:password@cluster.mongodb.net/indice_ditador?retryWrites=true&w=majority")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-search-preview"

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# PyObjectId class for handling MongoDB ObjectIds
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# Pydantic models
class ArticleModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    source: str
    url: str
    date: str
    score: float
    justification: str
    analysis_date: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Database connection
client = None
db = None


@app.on_event("startup")
async def startup_db_client():
    global client, db
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client.indice_ditador

    # Ensure collection and indexes exist
    await db.articles.create_index("analysis_date")

    # Test connection
    try:
        await client.admin.command('ping')
        print("Connected to MongoDB!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")


@app.on_event("shutdown")
async def shutdown_db_client():
    global client
    if client:
        client.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/crawl_news")
async def crawl_news(request: Request):
    """Crawl for recent Trump news related to democratic institutions"""
    data = await request.json()
    api_key = data.get("apiKey") or DEFAULT_OPENAI_API_KEY

    # Web search query for latest Trump news related to democratic institutions
    search_query = "recent news about Trump and democratic institutions today"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    search_payload = {
        "model": DEFAULT_MODEL,
        "web_search_options": {
            "search_context_size": "medium",
            "user_location": {
                "type": "approximate",
                "approximate": {
                    "country": "US",
                    "city": "Washington",
                    "region": "DC"
                }
            }
        },
        "messages": [{
            "role": "user",
            "content": "Find 3 recent news articles about Donald Trump that might relate to democratic institutions, checks and balances, or authoritarian tendencies. For each article, provide the title, source, publication date, and a brief summary of the content. Format your response as a clean JSON array with fields: title, source, url, date, and summary."
        }]
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=search_payload,
                headers=headers
            )

        search_results = response.json()

        if 'choices' in search_results and search_results['choices']:
            content = search_results['choices'][0]['message']['content']

            # Try to extract JSON from the response
            json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
            if json_match:
                articles_json = json.loads(json_match.group(0))
            else:
                # Fallback to processing the entire content as JSON
                try:
                    articles_json = json.loads(content)
                except:
                    return JSONResponse({"error": "Could not parse articles from search results"}, status_code=500)

            # Process each article
            analyzed_articles = []
            for article in articles_json:
                # Analyze article for authoritarianism
                analysis = await analyze_article(article, api_key)
                if analysis:
                    article_data = {
                        "title": article.get("title"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "date": article.get("date"),
                        "summary": article.get("summary"),
                        "score": float(analysis.get("score")),
                        "justification": analysis.get("justification"),
                        "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d")
                    }

                    analyzed_articles.append(article_data)

                    # Save to MongoDB
                    await db.articles.insert_one(article_data)

            # Calculate current index
            index_score = await calculate_index()

            return JSONResponse({
                "success": True,
                "articles": analyzed_articles,
                "index_score": index_score
            })
        else:
            return JSONResponse({"error": "Error in search API response", "details": search_results}, status_code=500)
    except Exception as e:
        print(f"Error in crawl_news: {str(e)}")
        return JSONResponse({"error": "Error processing request", "details": str(e)}, status_code=500)


async def analyze_article(article, api_key):
    """Analyze an article for authoritarian tendencies"""

    # Prepare analysis prompt
    analysis_prompt = f"""
    You are an expert political analyst specializing in democratic institutions and authoritarianism.

    Analyze the following news article regarding Donald Trump for signs of authoritarian tendencies:

    Title: {article.get('title')}
    Source: {article.get('source')}
    Date: {article.get('date')}
    Summary: {article.get('summary')}

    Rate the article on an authoritarianism scale from 0 to 10, where:
    - 0 means no authoritarian tendencies whatsoever
    - 10 means extreme authoritarian behavior that directly threatens democratic institutions

    Criteria to consider:
    1. Rhetoric attacking democratic institutions
    2. Threats to press freedom or political opponents
    3. Abuse of executive power
    4. Attempts to undermine judicial independence
    5. Disregard for democratic norms and processes

    Provide your numerical score and a brief justification (2-3 sentences).
    Format your response as valid JSON with two fields: "score" (number) and "justification" (string).
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    analysis_payload = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": analysis_prompt
        }]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=analysis_payload,
                headers=headers
            )

        result = response.json()

        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content']

            # Extract JSON from response
            try:
                json_match = re.search(r'{.*}', content, re.DOTALL)
                if json_match:
                    analysis_json = json.loads(json_match.group(0))
                    return analysis_json
                else:
                    print("No JSON found in analysis response")
                    return None
            except Exception as e:
                print(f"Error parsing analysis JSON: {str(e)}")
                return None
        else:
            print("Invalid analysis API response")
            return None
    except Exception as e:
        print(f"Error in analyze_article: {str(e)}")
        return None


async def calculate_index():
    """Calculate the current dictatorship index from database entries"""

    # Get all scores from MongoDB
    cursor = db.articles.find({}, {"score": 1})
    scores = []

    async for doc in cursor:
        scores.append(doc.get("score", 0))

    # Calculate average score if we have any scores
    if scores:
        average_score = sum(scores) / len(scores)
        return round(average_score, 1)
    else:
        return 0


@app.get("/get_index")
async def get_index():
    """Get the current dictatorship index and articles"""

    # Calculate current index score
    index_score = await calculate_index()

    # Get articles from MongoDB, sorted by analysis_date descending (most recent first)
    cursor = db.articles.find().sort("analysis_date", -1).limit(20)
    articles = []

    async for doc in cursor:
        articles.append({
            "title": doc.get("title"),
            "source": doc.get("source"),
            "url": doc.get("url"),
            "date": doc.get("date"),
            "score": float(doc.get("score")),
            "justification": doc.get("justification"),
            "analysis_date": doc.get("analysis_date")
        })

    return JSONResponse({
        "index_score": index_score,
        "articles": articles
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)