from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os
import json
import datetime
import re


# Read database/knowledge from files
def load_database():
    database_content = ""
    database_file = "codebase.txt"

    if os.path.exists(database_file):
        try:
            with open(database_file, 'r', encoding='utf-8') as f:
                database_content = f.read()
        except Exception as e:
            print(f"Error reading {database_file}: {str(e)}")

    return database_content


# Initialize variables
DATABASE = load_database()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-search-preview"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


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
                    analyzed_articles.append({
                        "title": article.get("title"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "date": article.get("date"),
                        "summary": article.get("summary"),
                        "authoritarianism_score": analysis.get("score"),
                        "justification": analysis.get("justification")
                    })

                    # Save to database
                    await save_to_database(article, analysis)

            # Calculate current index
            index_score = calculate_index()

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


async def save_to_database(article, analysis):
    """Save analyzed article to database"""
    global DATABASE

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    entry = f"""
=== Article ===
Title: {article.get('title')}
Source: {article.get('source')}
URL: {article.get('url')}
Date: {article.get('date')}
Score: {analysis.get('score')}
Justification: {analysis.get('justification')}
Analysis Date: {today}
=== End ===
"""

    DATABASE += entry

    # Write to file
    try:
        with open("codebase.txt", "w", encoding="utf-8") as f:
            f.write(DATABASE)
    except Exception as e:
        print(f"Error writing to database: {str(e)}")


def calculate_index():
    """Calculate the current dictatorship index from database entries"""
    global DATABASE

    scores = []
    score_pattern = r"Score: (\d+(?:\.\d+)?)"

    matches = re.finditer(score_pattern, DATABASE)
    for match in matches:
        try:
            score = float(match.group(1))
            scores.append(score)
        except:
            continue

    # Calculate average score if we have any scores
    if scores:
        average_score = sum(scores) / len(scores)
        return round(average_score, 1)
    else:
        return 0


@app.get("/get_index")
async def get_index():
    """Get the current dictatorship index and articles"""
    global DATABASE

    # Calculate current index score
    index_score = calculate_index()

    # Extract articles from database
    articles = []
    article_pattern = r"=== Article ===\s+Title: (.*?)\s+Source: (.*?)\s+URL: (.*?)\s+Date: (.*?)\s+Score: (.*?)\s+Justification: (.*?)\s+Analysis Date: (.*?)\s+=== End ==="

    matches = re.finditer(article_pattern, DATABASE, re.DOTALL)
    for match in matches:
        try:
            articles.append({
                "title": match.group(1).strip(),
                "source": match.group(2).strip(),
                "url": match.group(3).strip(),
                "date": match.group(4).strip(),
                "score": float(match.group(5).strip()),
                "justification": match.group(6).strip(),
                "analysis_date": match.group(7).strip()
            })
        except Exception as e:
            print(f"Error extracting article: {str(e)}")

    # Sort by analysis date, most recent first
    articles.sort(key=lambda x: x.get("analysis_date", ""), reverse=True)

    return JSONResponse({
        "index_score": index_score,
        "articles": articles[:20]  # Return the 20 most recent articles
    })