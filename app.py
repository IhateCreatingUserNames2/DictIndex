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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017") # Default to local if not set
DB_NAME = os.getenv("DB_NAME", "indice_ditador_db") # Specify your database name here or via env var
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model for search/summary tasks (can be cheaper if available with search)
# Replace with your actual cheaper model names if they are available and you've tested them
MODEL_FOR_SEARCH = os.getenv("MODEL_FOR_SEARCH", "gpt-4o-search-preview")
MODEL_FOR_ANALYSIS = os.getenv("MODEL_FOR_ANALYSIS", "gpt-4o")


# Number of items to fetch per crawl operation
NUM_ARTICLES_TO_FETCH_DITADOR = int(os.getenv("NUM_ARTICLES_TO_FETCH_DITADOR", 3))
NUM_NEWS_DYSTOPIAN = int(os.getenv("NUM_NEWS_DYSTOPIAN", 2))
NUM_SIM_TWEETS_DYSTOPIAN = int(os.getenv("NUM_SIM_TWEETS_DYSTOPIAN", 2))

MAX_TOKENS_FOR_ANALYSIS = int(os.getenv("MAX_TOKENS_FOR_ANALYSIS", 250))

# --- FastAPI Initialization ---
app = FastAPI(title="Índice de Ditador e Distopia API")

static_dir_exists = os.path.exists("static")
if static_dir_exists:
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("'static' directory not found. Creating it. Frontend might not load correctly until index.html is present.")
    os.makedirs("static", exist_ok=True)
    placeholder_html_path = "static/index.html"
    if not os.path.exists(placeholder_html_path):
        with open(placeholder_html_path, "w", encoding="utf-8") as f:
            f.write("<h1>Índice de Ditador & Distopia</h1><p>Placeholder content. Main HTML is missing or static files are not configured correctly.</p>")
        logger.info(f"Created placeholder '{placeholder_html_path}'.")


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

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, extra='ignore')

class DystopianMediaItemModel(BaseModel): # For "Índice de Distopia"
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    platform: str
    content_identifier: str
    title: Optional[str] = None
    content_text: str
    source_name: Optional[str] = None
    publication_date: Optional[str] = None
    dystopian_score: float
    justification: str
    themes_detected: List[str] = []
    analysis_date: str

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, extra='ignore')

# --- Database Connection ---
client: Optional[AsyncIOMotorClient] = None
db = None # Will hold the Database object

@app.on_event("startup")
async def startup_db_client():
    global client, db
    logger.info(f"Attempting to connect to MongoDB at {MONGODB_URL} and use database '{DB_NAME}'...")
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        await client.admin.command('ping') # Verify connection to the server
        logger.info("Successfully connected to MongoDB server!")
        
        db = client[DB_NAME] # Explicitly get the database by name
        if db is None: # Should not happen if client is valid and DB_NAME is a string
            logger.error(f"Failed to get database object for '{DB_NAME}'.")
            return
        logger.info(f"Using database: '{DB_NAME}'")

        # Collections and Indexes for "Índice de Ditador"
        logger.info("Ensuring indexes for 'articles' collection...")
        await db.articles.create_index("analysis_date")
        # IMPORTANT: If this next line fails with a duplicate key error,
        # you MUST manually clean your 'articles' collection in MongoDB
        # to remove documents with duplicate 'url' values.
        await db.articles.create_index("url", unique=True)
        logger.info("Indexes ensured for 'articles' collection.")

        # Collections and Indexes for "Índice de Distopia"
        logger.info("Ensuring indexes for 'dystopian_media_items' collection...")
        await db.dystopian_media_items.create_index("analysis_date")
        # IMPORTANT: Similar to above, ensure 'content_identifier' is unique in existing data.
        await db.dystopian_media_items.create_index("content_identifier", unique=True)
        logger.info("Indexes ensured for 'dystopian_media_items' collection.")

    except Exception as e:
        logger.error(f"Error during MongoDB startup or ensuring collections/indexes: {e}", exc_info=True)
        db = None # Ensure db is None if connection or setup fails

@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        logger.info("Closing MongoDB connection.")
        client.close()

# --- Helper Functions ---
async def _make_openai_call(http_client: httpx.AsyncClient, api_key: str, payload: Dict, call_description: str) -> Dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    logger.debug(f"Making OpenAI call for {call_description}. Model: {payload.get('model')}")
    try:
        response = await http_client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=90.0
        )
        response.raise_for_status()
        logger.debug(f"OpenAI call for {call_description} successful.")
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from OpenAI for {call_description}: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(f"Request error to OpenAI for {call_description}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in _make_openai_call for {call_description}: {e}", exc_info=True)
        raise

def _parse_llm_json_output(content: str, call_description: str, default_return: Any = None) -> Any:
    logger.debug(f"Attempting to parse LLM JSON output for {call_description}. Raw content (first 200 chars): {content[:200]}")
    try:
        # Attempt to find JSON array or object, potentially wrapped in markdown
        match = re.search(r'```json\s*([\s\S]*?)\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            logger.debug(f"Extracted JSON from markdown for {call_description}.")
            return json.loads(json_str)
        
        # If not in markdown, try to parse directly. LLM might return "articles": [...] or just [...]
        # This part needs to be robust based on how your prompts guide the LLM.
        # For now, assume it might be a direct JSON string.
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON output for {call_description}. Error: {e}. Content: {content[:500]}...")
        return default_return

# --- "Índice de Ditador" Logic ---
@app.post("/crawl_news")
async def crawl_news(request: Request):
    if db is None: return JSONResponse({"error": "Database not available. Check MongoDB connection and startup logs."}, status_code=503)
    req_data = await request.json()
    api_key = req_data.get("apiKey") or DEFAULT_OPENAI_API_KEY
    if not api_key: return JSONResponse({"error": "OpenAI API key is required."}, status_code=400)

    prompt_content = f"Find {NUM_ARTICLES_TO_FETCH_DITADOR} recent news articles (last 7 days) about Donald Trump that might relate to democratic institutions, checks and balances, or authoritarian tendencies. For each article, provide title, source, URL, publication date (YYYY-MM-DD), and a brief summary. Format your response as a JSON object with a single key 'articles' which contains an array of these article objects."
    search_payload = {
        "model": MODEL_FOR_SEARCH,
        "messages": [{"role": "user", "content": prompt_content}]
   
    }
    
    processed_articles_info = []
    newly_added_count = 0

    try:
        async with httpx.AsyncClient() as http_client:
            search_results_data = await _make_openai_call(http_client, api_key, search_payload, "Ditador Index News Search")
            
            content_str = search_results_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            parsed_response = _parse_llm_json_output(content_str, "Ditador Index News Search")
            articles_from_api = parsed_response.get("articles", []) if isinstance(parsed_response, dict) else []

            if not articles_from_api:
                logger.warning("No articles found or failed to parse from LLM search response for Ditador Index.")

            for article_data_api in articles_from_api:
                if not isinstance(article_data_api, dict):
                    logger.warning(f"Skipping non-dict item in articles_from_api: {article_data_api}")
                    continue
                url = article_data_api.get("url")
                if not url:
                    logger.info(f"Skipping article due to missing URL: {article_data_api.get('title', 'N/A')}")
                    continue
                if await db.articles.find_one({"url": url}):
                    logger.info(f"Article already exists (URL: {url}), skipping: {article_data_api.get('title', 'N/A')}")
                    continue

                analysis = await _analyze_article_authoritarianism(http_client, api_key, article_data_api)
                if not analysis: 
                    logger.warning(f"Failed to analyze article or analysis incomplete for: {article_data_api.get('title', 'N/A')}")
                    continue
                
                try:
                    article_obj = ArticleModel(
                        title=article_data_api.get("title", "N/A Title"),
                        source=article_data_api.get("source", "N/A Source"),
                        url=url,
                        date=article_data_api.get("date", datetime.date.today().isoformat()),
                        summary=article_data_api.get("summary"),
                        score=float(analysis["score"]),
                        justification=analysis["justification"],
                        analysis_date=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    )
                    db_data = article_obj.model_dump(by_alias=True, exclude_none=True)
                    insert_result = await db.articles.insert_one(db_data)
                    if insert_result.inserted_id:
                        newly_added_count += 1
                        response_article = article_obj.model_dump() # Use model_dump for response
                        response_article["id"] = str(insert_result.inserted_id)
                        processed_articles_info.append(response_article)
                except ValidationError as e:
                    logger.error(f"Pydantic validation error for article '{article_data_api.get('title')}': {e}", exc_info=True)
                except Exception as e: # Catch potential duplicate key error if somehow missed by find_one (race condition, unlikely)
                    logger.error(f"Error inserting article '{article_data_api.get('title')}' to DB: {e}", exc_info=True)

        index_score = await calculate_dictator_index()
        return JSONResponse({
            "success": True, "message": f"Dictator Index crawl complete. {newly_added_count} new articles added.",
            "articles_processed_this_crawl": processed_articles_info, "index_score": index_score
        })
    except Exception as e:
        logger.error(f"Error in crawl_news endpoint: {e}", exc_info=True)
        return JSONResponse({"error": "Error processing crawl_news request", "details": str(e)}, status_code=500)

async def _analyze_article_authoritarianism(http_client: httpx.AsyncClient, api_key: str, article_details: dict) -> Optional[Dict]:
    prompt = f"""Analyze the news article for authoritarian tendencies:
    Title: {article_details.get('title')}
    Summary: {article_details.get('summary')}
    Rate on a scale 0-10 (0=none, 10=extreme authoritarianism). Criteria: attacks on institutions, press freedom threats, abuse of power, undermining judiciary, disregard for democratic norms.
    Respond with JSON: {{"score": <number_float_or_int>, "justification": "<string_2_sentences_max>"}}"""
    payload = {"model": MODEL_FOR_ANALYSIS, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS_FOR_ANALYSIS}
    
    try:
        response_data = await _make_openai_call(http_client, api_key, payload, f"Authoritarianism Analysis for '{article_details.get('title')}'")
        content_str = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        analysis = _parse_llm_json_output(content_str, f"Authoritarianism Analysis for '{article_details.get('title')}'")
        if analysis and isinstance(analysis.get("score"), (int, float)) and isinstance(analysis.get("justification"), str):
            return analysis
        logger.warning(f"Invalid analysis format for authoritarianism: {analysis}. Article: {article_details.get('title')}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing article for authoritarianism '{article_details.get('title')}': {e}", exc_info=True)
        return None

async def calculate_dictator_index() -> float:
    if db is None: return 0.0
    pipeline = [
        {"$match": {"score": {"$exists": True, "$type": "double"}}},
        {"$group": {"_id": None, "average_score": {"$avg": "$score"}, "count": {"$sum": 1}}}
    ]
    try:
        result = await db.articles.aggregate(pipeline).to_list(length=1)
        return round(result[0]['average_score'], 1) if result and result[0].get('count',0) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error calculating dictator index: {e}", exc_info=True)
        return 0.0

@app.get("/get_index")
async def get_index():
    if db is None: return JSONResponse({"error": "Database not available."}, status_code=503)
    try:
        index_score = await calculate_dictator_index()
        articles_cursor = db.articles.find().sort("analysis_date", -1).limit(20)
        articles_list = []
        async for doc in articles_cursor:
            try:
                articles_list.append(ArticleModel.model_validate(doc).model_dump(exclude_none=True))
            except ValidationError as e:
                logger.warning(f"Validation error for article from DB (ID: {doc.get('_id')}): {e}")
        return JSONResponse({"index_score": index_score, "articles": articles_list})
    except Exception as e:
        logger.error(f"Error in get_index endpoint: {e}", exc_info=True)
        return JSONResponse({"error": "Error retrieving index data"}, status_code=500)


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
        news_prompt_content = f"Find {NUM_NEWS_DYSTOPIAN} recent news articles (last 7 days) discussing dystopian themes like mass surveillance, disinformation, loss of privacy, extreme social control, or unchecked AI. For each, provide title, source, URL, publication date (YYYY-MM-DD), and a brief summary. Format as a JSON object with a single key 'news_articles' which contains an array of these article objects."
        news_payload = {"model": MODEL_FOR_SEARCH, "messages": [{"role": "user", "content": news_prompt_content}] }
        
        try:
            news_search_data = await _make_openai_call(http_client, api_key, news_payload, "Dystopian News Search")
            news_content_str = news_search_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            parsed_news_response = _parse_llm_json_output(news_content_str, "Dystopian News Search")
            dystopian_news_from_api = parsed_news_response.get("news_articles", []) if isinstance(parsed_news_response, dict) else []

            for news_item_api in dystopian_news_from_api:
                if not isinstance(news_item_api, dict): 
                    logger.warning(f"Skipping non-dict item in dystopian_news_from_api: {news_item_api}")
                    continue
                url = news_item_api.get("url")
                if not url:
                    logger.info(f"Skipping dystopian news due to missing URL: {news_item_api.get('title', 'N/A')}")
                    continue
                if await db.dystopian_media_items.find_one({"content_identifier": url}):
                    logger.info(f"Dystopian news already exists (URL: {url}), skipping: {news_item_api.get('title', 'N/A')}")
                    continue
                
                analysis = await _analyze_media_for_dystopia(http_client, api_key, news_item_api, "Notícia Web")
                if not analysis: 
                    logger.warning(f"Failed to analyze dystopian news: {news_item_api.get('title', 'N/A')}")
                    continue

                item_obj = DystopianMediaItemModel(
                    platform="Notícia Web", content_identifier=url,
                    title=news_item_api.get("title", "N/A Title"),
                    content_text=news_item_api.get("summary", "N/A summary"),
                    source_name=news_item_api.get("source", "N/A source"),
                    publication_date=news_item_api.get("date", datetime.date.today().isoformat()),
                    dystopian_score=float(analysis["dystopian_score"]),
                    justification=analysis["justification"],
                    themes_detected=analysis.get("themes_detected", []),
                    analysis_date=datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                db_data = item_obj.model_dump(by_alias=True, exclude_none=True)
                insert_result = await db.dystopian_media_items.insert_one(db_data)
                if insert_result.inserted_id:
                    newly_added_count += 1
                    response_item = item_obj.model_dump()
                    response_item["id"] = str(insert_result.inserted_id)
                    processed_items_info.append(response_item)
        except Exception as e:
            logger.error(f"Error fetching/processing dystopian news: {e}", exc_info=True)

        # 2. Simulate Fetching "Tweets" for Dystopian Themes
        sim_tweets_prompt_content = f"Generate {NUM_SIM_TWEETS_DYSTOPIAN} plausible, recent-sounding (last 7 days) example social media posts (like tweets) expressing concern or observations about dystopian societal trends (e.g., surveillance, AI ethics, censorship). For each, provide 'simulated_user' (string), 'post_text' (string, concise like a tweet), 'simulated_date' (string YYYY-MM-DD). Format as a JSON object with a single key 'simulated_posts' which contains an array of these post objects."
        sim_tweets_payload = {"model": MODEL_FOR_ANALYSIS, "messages": [{"role": "user", "content": sim_tweets_prompt_content}] }
        
        try:
            sim_tweets_data = await _make_openai_call(http_client, api_key, sim_tweets_payload, "Simulated Dystopian Posts Generation")
            sim_tweets_content_str = sim_tweets_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            parsed_sim_tweets_response = _parse_llm_json_output(sim_tweets_content_str, "Simulated Dystopian Posts Generation")
            simulated_posts_from_api = parsed_sim_tweets_response.get("simulated_posts", []) if isinstance(parsed_sim_tweets_response, dict) else []

            for post_data_api in simulated_posts_from_api:
                if not isinstance(post_data_api, dict):
                    logger.warning(f"Skipping non-dict item in simulated_posts_from_api: {post_data_api}")
                    continue
                post_text = post_data_api.get("post_text")
                if not post_text:
                    logger.info(f"Skipping simulated post due to missing text.")
                    continue
                
                content_id = f"sim_post_{ObjectId()}" # Generate unique ID
                if await db.dystopian_media_items.find_one({"content_identifier": content_id}):
                    logger.info(f"Simulated post with ID {content_id} somehow already exists, skipping (very rare).")
                    continue

                analysis_input = {"summary": post_text, "title": f"Simulated Post by {post_data_api.get('simulated_user', 'Unknown User')}"}
                analysis = await _analyze_media_for_dystopia(http_client, api_key, analysis_input, "Tweet Simulado")
                if not analysis: 
                    logger.warning(f"Failed to analyze simulated post: {post_text[:50]}")
                    continue
                
                item_obj = DystopianMediaItemModel(
                    platform="Tweet Simulado", content_identifier=content_id,
                    content_text=post_text,
                    source_name=post_data_api.get("simulated_user", "SimulatedUser"),
                    publication_date=post_data_api.get("simulated_date", datetime.date.today().isoformat()),
                    dystopian_score=float(analysis["dystopian_score"]),
                    justification=analysis["justification"],
                    themes_detected=analysis.get("themes_detected", []),
                    analysis_date=datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                db_data = item_obj.model_dump(by_alias=True, exclude_none=True)
                insert_result = await db.dystopian_media_items.insert_one(db_data)
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
    item_title = item_details.get('title', 'N/A')
    prompt = f"""You are a sociologist specializing in dystopian trends. Analyze the following media item from '{platform_type}':
    Title/Identifier: {item_title}
    Content: {content_for_analysis}
    Rate this item on a scale of 0-10 for dystopian indicators (0=none, 10=strong).
    Criteria: surveillance, disinformation, loss of privacy, social control, suppression of freedom, unchecked tech power, extreme polarization, erosion of rights.
    Provide your score, a brief justification (1-2 sentences), and up to 3 main themes detected (from criteria or other relevant).
    Respond with JSON: {{"dystopian_score": <number_float_or_int>, "justification": "<string_2_sentences_max>", "themes_detected": ["<theme1>", "<theme2>"]}}"""
    
    payload = {"model": MODEL_FOR_ANALYSIS, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS_FOR_ANALYSIS}
    
    try:
        response_data = await _make_openai_call(http_client, api_key, payload, f"Dystopian Analysis for '{item_title}' ({platform_type})")
        content_str = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        analysis = _parse_llm_json_output(content_str, f"Dystopian Analysis for '{item_title}' ({platform_type})")
        if (analysis and isinstance(analysis.get("dystopian_score"), (int, float)) and
            isinstance(analysis.get("justification"), str) and isinstance(analysis.get("themes_detected"), list)):
            return analysis
        logger.warning(f"Invalid analysis format for dystopia: {analysis}. Item: {item_title}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing media for dystopia '{item_title}': {e}", exc_info=True)
        return None

async def calculate_dystopia_index() -> float:
    if db is None: return 0.0
    pipeline = [
        {"$match": {"dystopian_score": {"$exists": True, "$type": "double"}}},
        {"$group": {"_id": None, "average_score": {"$avg": "$dystopian_score"}, "count": {"$sum": 1}}}
    ]
    try:
        result = await db.dystopian_media_items.aggregate(pipeline).to_list(length=1)
        return round(result[0]['average_score'], 1) if result and result[0].get('count',0) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error calculating dystopia index: {e}", exc_info=True)
        return 0.0

@app.get("/get_dystopia_index_data")
async def get_dystopia_index_data_endpoint():
    if db is None: return JSONResponse({"error": "Database not available."}, status_code=503)
    try:
        index_score = await calculate_dystopia_index()
        items_cursor = db.dystopian_media_items.find().sort("analysis_date", -1).limit(20)
        items_list = []
        async for doc in items_cursor:
            try:
                items_list.append(DystopianMediaItemModel.model_validate(doc).model_dump(exclude_none=True))
            except ValidationError as e:
                 logger.warning(f"Validation error for dystopian item from DB (ID: {doc.get('_id')}): {e}")
        return JSONResponse({"dystopia_index_score": index_score, "items": items_list})
    except Exception as e:
        logger.error(f"Error in get_dystopia_index_data endpoint: {e}", exc_info=True)
        return JSONResponse({"error": "Error retrieving dystopian index data"}, status_code=500)

# --- Root and Static Files ---
@app.get("/", response_class=HTMLResponse)
async def root_get(request: Request): # Explicitly named for clarity
    # Check if the static directory and index.html exist before trying to serve
    static_html_path = "static/index.html"
    if os.path.exists(static_html_path) and static_dir_exists: # static_dir_exists checked at app init
        with open(static_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    logger.error(f"Root path '/' called, but '{static_html_path}' not found or static files not mounted correctly.")
    return HTMLResponse(content="<h1>Índice de Ditador & Distopia</h1><p>Error: Main index.html not found. Please ensure 'static/index.html' exists and is correctly served.</p>", status_code=404)

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    # The `static` directory and placeholder `index.html` creation is handled at app initialization.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
