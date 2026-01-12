import os
import pickle
from typing import List, Dict, Any,Tuple,Optional
import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException,Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE="https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

app= FastAPI(title="Movie Recommendation API",version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#PATH AND GLOBAL VARIABLE
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

DF_PATH=os.path.join(BASE_DIR,"df.pkl")
INDICES_PATH=os.path.join(BASE_DIR,"indices.pkl")
TFIDF_MATRIX_PATH=os.path.join(BASE_DIR,"tfidf_matrix.pkl")
TFIDF_PATH=os.path.join(BASE_DIR,"tfidf.pkl")

df:Optional[pd.DataFrame]=None
indices_obj:Any=None
tfidf_matrix:Any=None
tfidf_obj:Any=None

TITLE_TO_IDX:Optional[Dict[str,int]]=None

#MODEL

class TMDBMovieCard(BaseModel):
    tmdb_id:int
    title:str
    poster_url:Optional[str]=None
    release_date:Optional[str]=None
    vote_average:Optional[float]=None

class TMDBMovieDetails(BaseModel):
    tmdb_id:int
    title:str
    overview:Optional[str]=None
    release_date:Optional[str]=None
    poster_url:Optional[str]=None
    backdrop_url:Optional[str]=None
    genres:List[dict]=[]

class TFIDFRecItem(BaseModel):
    title:str
    score:float
    tmdb:Optional[TMDBMovieCard]=None

class SearchBundleResponse(BaseModel):
    query:str
    movie_details:TMDBMovieDetails
    tfidf_recommendations:List[TFIDFRecItem]
    genre_recommendations:List[TMDBMovieCard]    

#UTILITY FUNCTION
def _norm_title(t:str)->str:
    return str(t).strip().lower()


def make_img_url(path:Optional[str])->Optional[str]:
    if not path:
        return None        
    return f"{TMDB_IMG_500}{path}"

async def tmdb_get(path: str,params:Dict[str,Any])->Dict[str,Any]:

    """""
    safe TMDB GET:
    -Network errors ->502
    -TMDB API errors ->502 with details
    """

    if not TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB API key not configured")

    q=dict(params)
    q["api_key"]=TMDB_API_KEY

    try:
        async with httpx.AsyncClient() as client:
            r=await client.get(f"{TMDB_BASE}{path}",params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502,  detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",)

    if r.status_code!=200:
        raise HTTPException(status_code=502, detail=f"TMDB API error: {r.status_code} | {r.text}")
    return r.json()


async def tmdb_card_from_results(
        result:List[dict],limit:int=20

)->List[TMDBMovieCard]:
    out:List[TMDBMovieCard]=[]
    for m in (result or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out    


async def tmdb_movie_details(movie_id:int)->TMDBMovieDetails:
    data=await tmdb_get(f"/movie/{movie_id}",{"language":"en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres",[]) or [],

    )

async def tmdb_search_movies(query: str,page:int=1)->Dict[str,Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE result).
    Streamlit will use this for suggestion and grid
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query":query,
            "include_adult":"false",
            "language":"en-US",
            "page":page,
        },
    )


async def tmdb_search_first(query:str)->Optional[dict]:
    data=await tmdb_search_movies(query=query,page=1)
    results=data.get("results",[])
    return results[0] if results else None


def build_title_to_idx_map(indices:Any)->Dict[str,int]:
    """ indices.pkl can be:
    -dict(title->index)
    -pandas series (index=title,value=index)
    we normalize into title_to_idx
    """
    title_to_idx:Dict[str,int]={}

    if isinstance(indices,dict):
        for k,v in indices.items():
             title_to_idx[_norm_title(k)]=int(v)
        return title_to_idx

    #pandas series or similar mapping
    try:
        for k,v in indices.items():
            title_to_idx[_norm_title(k)]=int(v)
        return title_to_idx
    except Exception:
        #last resort: if it's list-like etc
        raise RuntimeError(
            "indices.pkl must be dict or pandas series-like (with .items())"

        )     
         
def get_local_idx_by_title(title:str)->int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500,detail="TF-IDF index map not initialized")
    key=_norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404,detail=f"Title not found in local dataset:'{title}'"
    )


def tfidf_recommend_title(
        query_title:str,top_n:int=10
)->List[Tuple[str,float]]:
    """ Return list of (title,score) from local df using cosine similarity on tf-idf matrix.
    safe against missing columns/rows.
    """

    global df,tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500,detail="TF-IDE resources not loaded")

    idx=get_local_idx_by_title(query_title)

    #query vector
    qv=tfidf_matrix[idx]
    scores=(tfidf_matrix @qv.T).toarray().ravel()

    #sort descending
    order =np.argsort(-scores)

    out:List[Tuple[str,float]]=[]
    for i in order:
        if int(i)==int(idx):
            continue

        try:
            title_i=str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i,float(scores[int(i)])))
        if len(out)>=top_n:
            break
    return out


async def attach_tmdb_card_by_title(title:str)->Optional[TMDBMovieCard]:
    """
    users tmdb search by title to fetch poster for a local title.
    if not found return none (never crashes the endpoint).
    """
    try:
        m=await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None
    
#starup:load pickles
@app.on_event("startup")
def load_pickles():
    global df, indices_obj,tfidf_matrix,tfidf_obj,TITLE_TO_IDX

    try:
        #load df
        with open(DF_PATH,"rb") as f:
            df=pickle.load(f)
    except Exception as e:
        print(f"Failed to load df.pkl: {e}")
        df = None

    try:
        #load indices
        with open(INDICES_PATH,"rb") as f:
            indices_obj=pickle.load(f)
    except Exception as e:
        print(f"Failed to load indices.pkl: {e}")
        indices_obj = None

    try:
        #load tf-idf matrix(usually scipy sparse)
        with open(TFIDF_MATRIX_PATH,"rb") as f:
              tfidf_obj =pickle.load(f)
        tfidf_matrix = tfidf_obj  # assuming tfidf_obj is the matrix
    except Exception as e:
        print(f"Failed to load tfidf_matrix.pkl: {e}")
        tfidf_matrix = None

    try:
         #build normalized map
        TITLE_TO_IDX=build_title_to_idx_map(indices_obj)
    except Exception as e:
        print(f"Failed to build title map: {e}")
        TITLE_TO_IDX = None

    # sanity
    if df is not None and ("title" not in df.columns):
        print("Warning: df.pkl does not contain a 'title' column")


#routes     
@app.get("/health")
def health():
    return {"status":"ok"}

#home route
# @app.get("/home",response_model=List[TMDBMovieCard])
# async def home(
#     category:str=Query("popular"),
#     limit:int=Query(24,ge=1,le=50),
# ):
#     try:
#         if category =="trending":
#             data =await tmdb_get("/trending/movie/day",{"language":"en-US"})
#             return await tmdb_card_from_results(data.get("result",[]),limit=limit)      
        
#         if category not in {"popular","top_rated","upcoming","now_playing"}:
#             raise HTTPException(status_code=400,detail="Invalid category")
        
#         data=await tmdb_get(f"/movie/{category}",{"language":"en-US","page":1})
#         return await tmdb_card_from_results(data.get("result",[]),limit=limit)
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500,detail=f"Home route failed:{e}")
# 
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    if not TMDB_API_KEY:
        return []  # Return empty list if no API key

    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_card_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_card_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")

    
#search multiple via keyword
@app.get("/tmdb/search")
async def tmdb_search(
    query:str=Query(...,min_length=1),
    page:int=Query(1,ge=1,le=10),
):
    """
    Returns RAW TMDB shape with 'result' list.
    Streamlit will use it for:
    -dropdown suggestions
    -grid results
    """
    return await tmdb_search_movies(query=query,page=page)

#movie details
@app.get("/movie/id/{tmdb_id}",response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id:int):
    return await tmdb_movie_details(tmdb_id)

#genre recommendation
@app.get("/recommend/genre",response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id:int=Query(...),
    limit:int=Query(18,ge=1,le=50),
):
    """
    Give a tmdb movie id:
    -fetch details
    -pickk first genre
    -discover movie in that genre (popular)
    """
    details=await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    
    genre_id=details.genres[0]["id"]
    discover=await tmdb_get(
        "/discover/movie",{
            "with_genres":genre_id,
            "language":"en-US",
            "sort_by":"popularity.desc",
            "page":1,
        },
    )
    cards=await tmdb_card_from_results(discover.get("results",[]),limit=limit)
    return [c for c in cards if c.tmdb_id !=tmdb_id]

#tf-idf only
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title:str=Query(...,min_length=1),
    top_n:int=Query(10,ge=1,le=50),
):
    recs=tfidf_recommend_title(title,top_n=top_n)
    return [{"title":t,"score":s} for t, s in recs]

#bundle : detail+tf-idf ,recs + genre res
@app.get("/movie/search",response_model=SearchBundleResponse)
async def search_bundle(
    query:str=Query(...,min_length=1),
    tfidf_top_n:int=Query(12,ge=1,le=30),
    genre_limit:int=Query(12,ge=1,le=30),
):
    
    """ this endpoint is for when you have a selected movie and want:
    -movie details
    -tf-idf recommendations (local)+posters
    -genre recommendations(tmdb)+posters
    
    NOTE:
    -it selects the best match from tmdb for the given query
    -if you want multiple mathes, use /tmdb/search
    """
    best=await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404,detail=f"No TMDB movie found for query:{query}"

        )
    tmdb_id=int(best["id"]) 
    details=await tmdb_movie_details(tmdb_id)

     # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_title(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_title(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_card_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )