# app.py

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
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from typing import List, Optional, Any, Dict
from bson import ObjectId
import logging

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/indice_ditador_db") # Default to local if not set
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model for search/summary tasks (can be cheaper if available with search)
MODEL_FOR_SEARCH = "gpt-4o-search-preview" # Or "gpt-4o" if using a generic search prompt
# Model for detailed analysis (prioritize quality)
MODEL_FOR_ANALYSIS = "gpt-4.1-nano-2025-04-14"

# Number of items to fetch per crawl operation
NUM_ARTICLES_TO_FETCH_DITADOR = 3
NUM_NEWS_DYSTOPIAN = 2 # Number of news articles for dystopian index
NUM_SIM_TWEETS_DYSTOPIAN = 2 # Number of simulated tweets for dystopian index

# Max tokens for analysis completion to control cost
MAX_TOKENS_FOR_ANALYSIS = 250

# --- FastAPI Initialization ---
app = FastAPI(title="Índice de Ditador e Distopia API")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("'static' directory not found. Frontend might not load.")
    os.makedirs("static", exist_ok=True) # Create if missing
    # Create a placeholder index.html if it's also missing
    if not os.path.exists("static/index.html"):
        with open("static/index.html", "w", encoding="utf-8") as f:
            f.write("<h1>Índice de Ditador & Distopia</h1><p>Placeholder content. Main HTML is missing.</p>")
        logger.info("Created placeholder 'static/index.html'.")


# --- Pydantic Models & Custom Types ---
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any, _handler: Any = None) -> str:
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str):
            if ObjectId.is_valid(v):
                return v
            raise ValueError(f"'{v}' is not a valid ObjectId string")
        raise TypeError(f"ObjectId or valid ObjectId string expected, got {type(v).__name__}")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string", "format": "objectid-string"}

class ArticleModel(BaseModel): # For "Índice de Ditador"
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    title: str
    source: str
    url: str
    date: str
    summary: Optional[str] = None
    score: float
    justification: str
    analysis_date: str

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

class DystopianMediaItemModel(BaseModel): # For "Índice de Distopia"
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    platform: str  # Ex: "Notícia Web", "Tweet Simulado"
    content_identifier: str # Ex: URL for news, generated ID for simulated tweet
    title: Optional[str] = None # For news articles
    content_text: str # Main text for analysis (summary for news, tweet text for tweets)
    source_name: Optional[str] = None # e.g., "New York Times", "Simulated Twitter User"
    publication_date: Optional[str] = None
    dystopian_score: float
    justification: str
    themes_detected: List[str] = []
    analysis_date: str

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

# --- Database Connection ---
client: Optional[AsyncIOMotorClient] = None
db = None

@app.on_event("startup")
async def startup_db_client():
    global client, db
    logger.info("Attempting to connect to MongoDB...")
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        db = client.get_database() # Gets default DB from URI or you can specify name

        # Collections and Indexes for "Índice de Ditador"
        await db.articles.create_index("analysis_date")
        await db.articles.create_index("url", unique=True)
        logger.info("Indexes ensured for 'articles' collection.")

        # Collections and Indexes for "Índice de Distopia"
        await db.dystopian_media_items.create_index("analysis_date")
        await db.dystopian_media_items.create_index("content_identifier", unique=True)
        logger.info("Indexes ensured for 'dystopian_media_items' collection.")

    except Exception as e:
        logger.error(f"Error connecting to MongoDB or ensuring collections/indexes: {e}", exc_info=True)
        db = None # Ensure db is None if connection fails

@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        logger.info("Closing MongoDB connection.")
        client.close()

# --- Helper Functions ---
async def _make_openai_call(http_client: httpx.AsyncClient, api_key: str, payload: Dict) -> Dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = await http_client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=90.0 # Increased timeout for potentially longer search/generation tasks
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from OpenAI: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(f"Request error to OpenAI: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in _make_openai_call: {e}", exc_info=True)
        raise

def _parse_llm_json_output(content: str, default_return: Any = None) -> Any:
    try:
        # Attempt to find JSON array or object
        match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*\])|({[\s\S]*})', content, re.DOTALL)
        if match:
            json_str = match.group(1) or match.group(2) or match.group(3)
            return json.loads(json_str)
        return json.loads(content) # Fallback if no explicit markers
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON output. Content: {content[:200]}...")
        return default_return

# --- "Índice de Ditador" Logic ---
@app.post("/crawl_news") # Existing endpoint for Ditador Index
async def crawl_news(request: Request):
    if db is None: return JSONResponse({"error": "Database not available."}, status_code=503)
    req_data = await request.json()
    api_key = req_data.get("apiKey") or DEFAULT_OPENAI_API_KEY
    if not api_key: return JSONResponse({"error": "OpenAI API key is required."}, status_code=400)

    search_payload = {
        "model": MODEL_FOR_SEARCH,
        "messages": [{"role": "user", "content": f"Find {NUM_ARTICLES_TO_FETCH_DITADOR} recent news articles (last 7 days) about Donald Trump that might relate to democratic institutions, checks and balances, or authoritarian tendencies. For each article, provide title, source, URL, publication date (YYYY-MM-DD), and a brief summary. Format your response as a clean JSON array."}],
        "response_format": {"type": "json_object"} # More reliable if model supports it well for arrays
    }
    # If response_format {"type": "json_object"} doesn't work well for arrays, remove it and parse manually.
    # Some models expect json_object to be an object, not an array.

    processed_articles_info = []
    newly_added_count = 0

    try:
        async with httpx.AsyncClient() as http_client:
            search_results_data = await _make_openai_call(http_client, api_key, search_payload)
            
            content_str = search_results_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            # Assuming the JSON object contains a key like "articles" which is an array
            # This depends on how you instruct the LLM or if you directly parse an array response
            articles_from_api = _parse_llm_json_output(content_str, {}).get("articles", [])
            if not isinstance(articles_from_api, list):
                 articles_from_api = _parse_llm_json_output(content_str, []) # try parsing as direct array

            if not articles_from_api:
                logger.warning("No articles found or failed to parse from LLM search response for Ditador Index.")
                # Fallback logic or error response might be needed here

            for article_data_api in articles_from_api:
                url = article_data_api.get("url")
                if not url or await db.articles.find_one({"url": url}):
                    logger.info(f"Skipping article (no URL or duplicate): {article_data_api.get('title', 'N/A')}")
                    continue

                analysis = await _analyze_article_authoritarianism(http_client, api_key, article_data_api)
                if not analysis: continue

                article_obj = ArticleModel(
                    title=article_data_api.get("title", "N/A"),
                    source=article_data_api.get("source", "N/A"),
                    url=url,
                    date=article_data_api.get("date", datetime.date.today().isoformat()),
                    summary=article_data_api.get("summary"),
                    score=float(analysis["score"]),
                    justification=analysis["justification"],
                    analysis_date=datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                insert_result = await db.articles.insert_one(article_obj.model_dump(by_alias=True, exclude_none=True))
                if insert_result.inserted_id:
                    newly_added_count += 1
                    response_article = article_obj.model_dump()
                    response_article["id"] = str(insert_result.inserted_id)
                    processed_articles_info.append(response_article)
        
        index_score = await calculate_dictator_index()
        return JSONResponse({
            "success": True, "message": f"Dictator Index crawl complete. {newly_added_count} new articles added.",
            "articles_processed_this_crawl": processed_articles_info, "index_score": index_score
        })
    except Exception as e:
        logger.error(f"Error in crawl_news: {e}", exc_info=True)
        return JSONResponse({"error": "Error processing crawl_news request", "details": str(e)}, status_code=500)

async def _analyze_article_authoritarianism(http_client: httpx.AsyncClient, api_key: str, article_details: dict) -> Optional[Dict]:
    prompt = f"""Analyze the news article for authoritarian tendencies:
    Title: {article_details.get('title')}
    Summary: {article_details.get('summary')}
    Rate on a scale 0-10 (0=none, 10=extreme authoritarianism). Criteria: attacks on institutions, press freedom threats, abuse of power, undermining judiciary, disregard for democratic norms.
    Respond with JSON: {{"score": <number>, "justification": "<string>"}}"""
    payload = {"model": MODEL_FOR_ANALYSIS, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS_FOR_ANALYSIS}
    
    try:
        response_data = await _make_openai_call(http_client, api_key, payload)
        content_str = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        analysis = _parse_llm_json_output(content_str)
        if analysis and isinstance(analysis.get("score"), (int, float)) and isinstance(analysis.get("justification"), str):
            return analysis
        logger.warning(f"Invalid analysis format for authoritarianism: {analysis}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing article for authoritarianism: {e}", exc_info=True)
        return None

async def calculate_dictator_index() -> float:
    if db is None: return 0.0
    pipeline = [
        {"$match": {"score": {"$exists": True, "$type": "double"}}}, # Ensure score exists and is double
        {"$group": {"_id": None, "average_score": {"$avg": "$score"}, "count": {"$sum": 1}}}
    ]
    result = await db.articles.aggregate(pipeline).to_list(length=1)
    return round(result[0]['average_score'], 1) if result and result[0].get('count',0) > 0 else 0.0

@app.get("/get_index") # Existing endpoint for Ditador Index
async def get_index():
    if db is None: return JSONResponse({"error": "Database not available."}, status_code=503)
    index_score = await calculate_dictator_index()
    articles_cursor = db.articles.find().sort("analysis_date", -1).limit(20)
    articles_list = [ArticleModel.model_validate(doc).model_dump(exclude_none=True) async for doc in articles_cursor]
    return JSONResponse({"index_score": index_score, "articles": articles_list})


# --- "Índice de Distopia" Logic ---
@app.post("/crawl_dystopian_content")
async def crawl_dystopian_content_endpoint(request: Request):
    if db is None: return JSONResponse({"error": "Database not available."}, status_code=503)
    req_data = await request.json()
    api_key = req_data.get("apiKey") or DEFAULT_OPENAI_API_KEY
    if not api_key: return JSONResponse({"error": "OpenAI API key is required."}, status_code=400)

    processed_items_info = []
    newly_added_count = 0

    async with httpx.AsyncClient() as http_client:
        # 1. Fetch News Articles for Dystopian Themes
        news_prompt = f"Find {NUM_NEWS_DYSTOPIAN} recent news articles (last 7 days) discussing dystopian themes like mass surveillance, disinformation, loss of privacy, extreme social control, or unchecked AI. For each, provide title, source, URL, publication date (YYYY-MM-DD), and a brief summary. Format as a JSON array within a parent JSON object under the key 'news_articles'."
        news_payload = {"model": MODEL_FOR_SEARCH, "messages": [{"role": "user", "content": news_prompt}], "response_format": {"type": "json_object"}}
        
        try:
            news_search_data = await _make_openai_call(http_client, api_key, news_payload)
            news_content_str = news_search_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            dystopian_news_from_api = _parse_llm_json_output(news_content_str, {}).get("news_articles", [])

            for news_item_api in dystopian_news_from_api:
                url = news_item_api.get("url")
                if not url or await db.dystopian_media_items.find_one({"content_identifier": url}):
                    logger.info(f"Skipping dystopian news (no URL or duplicate): {news_item_api.get('title', 'N/A')}")
                    continue
                
                analysis = await _analyze_media_for_dystopia(http_client, api_key, news_item_api, "Notícia Web")
                if not analysis: continue

                item_obj = DystopianMediaItemModel(
                    platform="Notícia Web",
                    content_identifier=url,
                    title=news_item_api.get("title", "N/A"),
                    content_text=news_item_api.get("summary", "N/A summary"),
                    source_name=news_item_api.get("source", "N/A source"),
                    publication_date=news_item_api.get("date", datetime.date.today().isoformat()),
                    dystopian_score=float(analysis["dystopian_score"]),
                    justification=analysis["justification"],
                    themes_detected=analysis.get("themes_detected", []),
                    analysis_date=datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                insert_result = await db.dystopian_media_items.insert_one(item_obj.model_dump(by_alias=True, exclude_none=True))
                if insert_result.inserted_id:
                    newly_added_count += 1
                    response_item = item_obj.model_dump()
                    response_item["id"] = str(insert_result.inserted_id)
                    processed_items_info.append(response_item)
        except Exception as e:
            logger.error(f"Error fetching/processing dystopian news: {e}", exc_info=True)


        # 2. Simulate Fetching "Tweets" for Dystopian Themes
        sim_tweets_prompt = f"Generate {NUM_SIM_TWEETS_DYSTOPIAN} plausible, recent-sounding (last 7 days) example social media posts (like tweets) expressing concern or observations about dystopian societal trends (e.g., surveillance, AI ethics, censorship). For each, provide 'simulated_user' (string), 'post_text' (string), 'simulated_date' (string YYYY-MM-DD). Format as a JSON array within a parent JSON object under the key 'simulated_posts'."
        sim_tweets_payload = {"model": MODEL_FOR_ANALYSIS, "messages": [{"role": "user", "content": sim_tweets_prompt}], "response_format": {"type": "json_object"}} # Use analysis model as it's good at generation
        
        try:
            sim_tweets_data = await _make_openai_call(http_client, api_key, sim_tweets_payload)
            sim_tweets_content_str = sim_tweets_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            simulated_posts_from_api = _parse_llm_json_output(sim_tweets_content_str, {}).get("simulated_posts", [])

            for post_data_api in simulated_posts_from_api:
                post_text = post_data_api.get("post_text")
                # Generate a unique identifier for simulated content
                content_id = f"sim_post_{ObjectId()}" 
                if not post_text or await db.dystopian_media_items.find_one({"content_identifier": content_id}): # Very unlikely to clash, but good practice
                    logger.info(f"Skipping simulated post (no text or rare duplicate ID): {post_text[:50]}")
                    continue

                # Adapt post_data_api for analysis (it's already the content itself)
                analysis_input = {
                    "summary": post_text, # Use summary field for analysis prompt consistency
                    "title": f"Simulated Post by {post_data_api.get('simulated_user', 'Unknown User')}"
                }
                analysis = await _analyze_media_for_dystopia(http_client, api_key, analysis_input, "Tweet Simulado")
                if not analysis: continue
                
                item_obj = DystopianMediaItemModel(
                    platform="Tweet Simulado",
                    content_identifier=content_id,
                    title=None, # No traditional title for a tweet
                    content_text=post_text,
                    source_name=post_data_api.get("simulated_user", "SimulatedUser"),
                    publication_date=post_data_api.get("simulated_date", datetime.date.today().isoformat()),
                    dystopian_score=float(analysis["dystopian_score"]),
                    justification=analysis["justification"],
                    themes_detected=analysis.get("themes_detected", []),
                    analysis_date=datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                insert_result = await db.dystopian_media_items.insert_one(item_obj.model_dump(by_alias=True, exclude_none=True))
                if insert_result.inserted_id:
                    newly_added_count += 1
                    response_item = item_obj.model_dump()
                    response_item["id"] = str(insert_result.inserted_id)
                    processed_items_info.append(response_item)
        except Exception as e:
            logger.error(f"Error generating/processing simulated dystopian posts: {e}", exc_info=True)

    dystopia_index_score = await calculate_dystopia_index()
    return JSONResponse({
        "success": True, "message": f"Dystopian Index crawl complete. {newly_added_count} new items added.",
        "items_processed_this_crawl": processed_items_info, "dystopia_index_score": dystopia_index_score
    })

async def _analyze_media_for_dystopia(http_client: httpx.AsyncClient, api_key: str, item_details: dict, platform_type: str) -> Optional[Dict]:
    content_for_analysis = item_details.get('summary') or item_details.get('content_text') or item_details.get('post_text')
    prompt = f"""You are a sociologist specializing in dystopian trends. Analyze the following media item from '{platform_type}':
    Title/Identifier: {item_details.get('title', 'N/A')}
    Content: {content_for_analysis}
    Rate this item on a scale of 0-10 for dystopian indicators (0=none, 10=strong).
    Criteria: surveillance, disinformation, loss of privacy, social control, suppression of freedom, unchecked tech power, extreme polarization, erosion of rights.
    Provide your score, a brief justification (1-2 sentences), and up to 3 main themes detected.
    Respond with JSON: {{"dystopian_score": <number>, "justification": "<string>", "themes_detected": ["<theme1>", "<theme2>"]}}"""
    
    payload = {"model": MODEL_FOR_ANALYSIS, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS_FOR_ANALYSIS}
    
    try:
        response_data = await _make_openai_call(http_client, api_key, payload)
        content_str = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        analysis = _parse_llm_json_output(content_str)
        if (analysis and isinstance(analysis.get("dystopian_score"), (int, float)) and
            isinstance(analysis.get("justification"), str) and isinstance(analysis.get("themes_detected"), list)):
            return analysis
        logger.warning(f"Invalid analysis format for dystopia: {analysis}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing media for dystopia: {e}", exc_info=True)
        return None

async def calculate_dystopia_index() -> float:
    if db is None: return 0.0
    pipeline = [
        {"$match": {"dystopian_score": {"$exists": True, "$type": "double"}}},
        {"$group": {"_id": None, "average_score": {"$avg": "$dystopian_score"}, "count": {"$sum": 1}}}
    ]
    result = await db.dystopian_media_items.aggregate(pipeline).to_list(length=1)
    return round(result[0]['average_score'], 1) if result and result[0].get('count',0) > 0 else 0.0

@app.get("/get_dystopia_index_data")
async def get_dystopia_index_data_endpoint():
    if db is None: return JSONResponse({"error": "Database not available."}, status_code=503)
    index_score = await calculate_dystopia_index()
    items_cursor = db.dystopian_media_items.find().sort("analysis_date", -1).limit(20)
    items_list = [DystopianMediaItemModel.model_validate(doc).model_dump(exclude_none=True) async for doc in items_cursor]
    return JSONResponse({"dystopia_index_score": index_score, "items": items_list})

# --- Root and Static Files ---
@app.get("/", response_class=HTMLResponse)
async def root():
    static_html_path = "static/index.html"
    if os.path.exists(static_html_path):
        with open(static_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Índice de Ditador & Distopia</h1><p>Error: Main index.html not found in /static directory.</p>", status_code=404)

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
