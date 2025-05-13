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
from typing import List, Optional, Any
from bson import ObjectId

# Load environment variables
load_dotenv()

# MongoDB connection string
MONGODB_URL = os.getenv("MONGODB_URL",
                        "mongodb+srv://username:password@cluster.mongodb.net/indice_ditador?retryWrites=true&w=majority")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL_SEARCH = "gpt-4o-search-preview" # Model for news searching
DEFAULT_MODEL_ANALYSIS = "gpt-4o" # Model for analysis

# Initialize FastAPI
app = FastAPI(title="Índice de Ditador API")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Custom ObjectId field
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any, _handler: Any = None) -> str: # Added _handler for Pydantic v2 compatibility
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str):
            if ObjectId.is_valid(v):
                return v
            else:
                raise ValueError(f"'{v}' is not a valid ObjectId string")
        raise TypeError(f"ObjectId or valid ObjectId string expected, got {type(v).__name__}")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler): # Pydantic v2 signature
        # For Pydantic v2, if you're using __get_validators__ (v1 style), 
        # this method might not be called or might need the old signature.
        # schema = handler(core_schema) if handler and core_schema else {}
        # schema.update(type="string", format="objectid")
        # return schema
        # Sticking to user's original simple version if it worked for them:
        # def __get_pydantic_json_schema__(cls, _schema_generator):
        return {"type": "string", "format": "objectid-string"}


# Pydantic models
class ArticleModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    title: str
    source: str
    url: str
    date: str # Consider validating format e.g. YYYY-MM-DD
    summary: Optional[str] = None # Added summary field
    score: float
    justification: str
    analysis_date: str # Consider validating format e.g. YYYY-MM-DD

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True, # Necessary for PyObjectId if not using CoreSchema
        json_schema_extra={
            "example": {
                "title": "Trump threatens to fire officials",
                "source": "Washington Post",
                "url": "https://example.com/article",
                "date": "2025-05-01",
                "summary": "A brief summary of the article content.",
                "score": 7.5,
                "justification": "Rhetoric undermining institutions",
                "analysis_date": "2025-05-02"
            }
        }
    )


# Database connection
client: Optional[AsyncIOMotorClient] = None
db = None


@app.on_event("startup")
async def startup_db_client():
    global client, db
    print("Attempting to connect to MongoDB...")
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        # Test connection by pinging
        await client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        db = client.indice_ditador # Use your database name

        # Ensure 'articles' collection and 'analysis_date' index exist
        print("Ensuring 'articles' collection and indexes...")
        if "articles" not in await db.list_collection_names():
            print("Creating 'articles' collection.")
        await db.articles.create_index("analysis_date")
        await db.articles.create_index("url", unique=True) # Add unique index on URL to enforce uniqueness at DB level
        print("Indexes ensured on 'articles' collection (analysis_date, url).")

    except Exception as e:
        print(f"Error connecting to MongoDB or ensuring collections/indexes: {e}")
        # client might be None or partially initialized.
        # Consider how the app should behave if DB connection fails.
        # For now, it will proceed, and db operations will fail.


@app.on_event("shutdown")
async def shutdown_db_client():
    global client
    if client:
        print("Closing MongoDB connection.")
        client.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    # Ensure static/index.html exists
    if not os.path.exists("static/index.html"):
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Make sure the static/index.html file exists in your project directory.</p>", status_code=500)
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/crawl_news")
async def crawl_news(request: Request):
    """Crawl for recent Trump news, analyze, and store if new."""
    if db is None:
        return JSONResponse({"error": "Database not available. Check MongoDB connection."}, status_code=503)

    data = await request.json()
    api_key = data.get("apiKey") or DEFAULT_OPENAI_API_KEY
    if not api_key:
        return JSONResponse({"error": "OpenAI API key is required."}, status_code=400)

    search_query = "recent news about Trump and democratic institutions today"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    search_payload = {
        "model": DEFAULT_MODEL_SEARCH,
        "web_search_options": { # This option might be specific to certain OpenAI API versions or models
            "search_context_size": "medium",
            "user_location": {"type": "approximate", "approximate": {"country": "US", "city": "Washington", "region": "DC"}}
        },
        "messages": [{"role": "user", "content": "Find 3 recent news articles about Donald Trump that might relate to democratic institutions, checks and balances, or authoritarian tendencies. For each article, provide the title, source, publication date, URL, and a brief summary of the content. Format your response as a clean JSON array with fields: title, source, url, date, and summary."}]
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post("https://api.openai.com/v1/chat/completions", json=search_payload, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
        search_results = response.json()

        if not ('choices' in search_results and search_results['choices']):
            return JSONResponse({"error": "Invalid response from search API", "details": search_results}, status_code=500)

        content = search_results['choices'][0]['message']['content']
        articles_json_list = []
        try:
            # Try to extract JSON array from the response (handles cases where LLM wraps JSON in text)
            json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
            if json_match:
                articles_json_list = json.loads(json_match.group(0))
            else:
                articles_json_list = json.loads(content) # Assume content is a direct JSON array
            if not isinstance(articles_json_list, list): # Ensure it's a list
                raise ValueError("Parsed content is not a list of articles")
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse({"error": "Could not parse articles from search results", "details": str(e), "raw_content": content}, status_code=500)

        processed_articles_info = []
        newly_added_count = 0

        for article_data_from_api in articles_json_list:
            article_url = article_data_from_api.get("url")
            if not article_url:
                print(f"Skipping article due to missing URL: {article_data_from_api.get('title', 'N/A')}")
                continue

            # 1. Avoid Repetition: Check if article URL already exists
            existing_article = await db.articles.find_one({"url": article_url})
            if existing_article:
                print(f"Article already exists (URL: {article_url}), skipping.")
                # Optionally, add to a list of "skipped_duplicates" to inform the user
                continue

            analysis = await analyze_article(article_data_from_api, api_key, http_client)
            if not analysis or "score" not in analysis or "justification" not in analysis:
                print(f"Failed to analyze article or analysis incomplete: {article_data_from_api.get('title', 'N/A')}")
                continue
            
            current_analysis_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            
            article_payload = {
                "title": article_data_from_api.get("title"),
                "source": article_data_from_api.get("source"),
                "url": article_url,
                "date": article_data_from_api.get("date"), # Ensure this is a valid date string
                "summary": article_data_from_api.get("summary"), # Store summary
                "score": float(analysis["score"]),
                "justification": analysis["justification"],
                "analysis_date": current_analysis_date
            }

            try:
                article_to_save = ArticleModel(**article_payload)
            except ValidationError as e:
                print(f"Data validation error for article '{article_payload.get('title')}': {e}")
                continue
            
            # For insertion, Pydantic model ensures data types.
            # model_dump by_alias=True will use "_id" if id field is set.
            # Here, id is None, so it's excluded by exclude_none=True. MongoDB generates _id.
            db_data = article_to_save.model_dump(by_alias=True, exclude_none=True)
            
            try:
                insert_result = await db.articles.insert_one(db_data)
                if insert_result.inserted_id:
                    newly_added_count += 1
                    # Prepare data for response (can use article_to_save.model_dump() or manually construct)
                    # Using the validated and structured data
                    response_article_data = article_to_save.model_dump(exclude_none=True)
                    response_article_data["id"] = str(insert_result.inserted_id) # Add the new ID as string
                    processed_articles_info.append(response_article_data)
                else:
                    print(f"Failed to insert article: {article_to_save.title}")
            except Exception as e: # Catches potential duplicate key error if URL unique index is very strict
                print(f"Error inserting article '{article_to_save.title}' to DB: {e}")


        index_score = await calculate_index()
        return JSONResponse({
            "success": True,
            "message": f"Crawl complete. {newly_added_count} new articles added.",
            "articles_processed_this_crawl": processed_articles_info, # Articles newly added in this run
            "index_score": index_score
        })

    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": "HTTP error during OpenAI API call", "details": str(e), "request_url": str(e.request.url), "response_content": e.response.text if e.response else "No response content"}, status_code=e.response.status_code if e.response else 500)
    except httpx.RequestError as e:
        return JSONResponse({"error": "Request error during OpenAI API call", "details": str(e), "request_url": str(e.request.url)}, status_code=500)
    except Exception as e:
        print(f"Unexpected error in crawl_news: {str(e)}")
        return JSONResponse({"error": "An unexpected error occurred", "details": str(e)}, status_code=500)


async def analyze_article(article_details: dict, api_key: str, http_client: httpx.AsyncClient):
    """Analyze an article for authoritarian tendencies using OpenAI."""
    analysis_prompt = f"""
    You are an expert political analyst specializing in democratic institutions and authoritarianism.
    Analyze the following news article regarding Donald Trump for signs of authoritarian tendencies:

    Title: {article_details.get('title', 'N/A')}
    Source: {article_details.get('source', 'N/A')}
    Date: {article_details.get('date', 'N/A')}
    Summary: {article_details.get('summary', 'N/A')}

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
    Format your response as valid JSON with two fields: "score" (number, e.g., 7.5) and "justification" (string).
    Example: {{"score": 7.5, "justification": "The article details rhetoric undermining judicial independence."}}
    """

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    analysis_payload = {"model": DEFAULT_MODEL_ANALYSIS, "messages": [{"role": "user", "content": analysis_prompt}], "response_format": {"type": "json_object"}} # Request JSON output

    try:
        response = await http_client.post("https://api.openai.com/v1/chat/completions", json=analysis_payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        result = response.json()

        if 'choices' in result and result['choices']:
            content_str = result['choices'][0]['message']['content']
            try:
                # OpenAI with response_format: "json_object" should return valid JSON string directly
                analysis_json = json.loads(content_str)
                if isinstance(analysis_json.get("score"), (int, float)) and isinstance(analysis_json.get("justification"), str):
                    return analysis_json
                else:
                    print(f"Analysis JSON has incorrect types: {analysis_json}")
                    return None
            except json.JSONDecodeError:
                print(f"Error parsing analysis JSON from content: {content_str}")
                return None # Could attempt regex as fallback if needed
        else:
            print(f"Invalid analysis API response structure: {result}")
            return None
    except httpx.HTTPStatusError as e:
        print(f"HTTP error during analysis API call for article '{article_details.get('title', 'N/A')}': {e} - {e.response.text if e.response else ''}")
        return None
    except httpx.RequestError as e:
        print(f"Request error during analysis API call for article '{article_details.get('title', 'N/A')}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in analyze_article for '{article_details.get('title', 'N/A')}': {str(e)}")
        return None


async def calculate_index() -> float:
    """Calculate the current dictatorship index from all database entries."""
    if db is None or db.articles is None:
        return 0.0

    # Using MongoDB aggregation to calculate average score directly in the database.
    # This is more efficient for large datasets.
    # It assumes 'score' field exists and is numeric.
    # ArticleModel ensures 'score' is a float, so this should be safe.
    pipeline = [
        {"$match": {"score": {"$exists": True, "$type": "double"}}}, # Ensure score exists and is a double (float)
        {"$group": {"_id": None, "average_score": {"$avg": "$score"}, "count": {"$sum": 1}}}
    ]
    
    aggregation_result = await db.articles.aggregate(pipeline).to_list(length=1)

    if aggregation_result and aggregation_result[0].get("count", 0) > 0:
        avg_score = aggregation_result[0].get("average_score")
        return round(avg_score, 1) if avg_score is not None else 0.0
    else:
        # Fallback or if no articles with valid scores found
        # This also covers the case where the collection is empty.
        return 0.0


@app.get("/get_index")
async def get_index():
    """Get the current dictatorship index and recent articles."""
    if db is None:
        return JSONResponse({"error": "Database not available. Check MongoDB connection."}, status_code=503)

    index_score = await calculate_index()

    # Get articles from MongoDB, sorted by analysis_date descending, limit for display
    # The index calculation uses ALL articles; this is just for display.
    cursor = db.articles.find().sort("analysis_date", -1).limit(20)
    articles_list = []

    async for doc_from_db in cursor:
        try:
            # Validate and structure data using Pydantic model
            article_model = ArticleModel.model_validate(doc_from_db)
            # Convert to dict for JSON response (ensure _id is str via PyObjectId)
            articles_list.append(article_model.model_dump(exclude_none=True))
        except ValidationError as e:
            print(f"Data validation error for article from DB (ID: {doc_from_db.get('_id')}): {e}")
            # Optionally skip or include a partial/error representation

    return JSONResponse({
        "index_score": index_score,
        "articles": articles_list # This list uses Pydantic for structure
    })


if __name__ == "__main__":
    import uvicorn
    # Ensure 'static' directory exists for StaticFiles
    if not os.path.exists("static"):
        os.makedirs("static", exist_ok=True)
        print("Created 'static' directory as it was missing.")
        # You might want to create a placeholder index.html if it's also missing
        if not os.path.exists("static/index.html"):
            with open("static/index.html", "w", encoding="utf-8") as f:
                f.write("<h1>Índice de Ditador</h1><p>Frontend not fully initialized. Please add content to static/index.html.</p>")
            print("Created placeholder 'static/index.html'.")

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
