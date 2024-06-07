import os
import re

from typing import List
from pathlib import Path

import redis
import pandas as pd

from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, FastAPI, Query, HTTPException, status
from prometheus_client import make_wsgi_app, Summary, Counter


def read(file_name):
    """ # Read File
    This function reads the contents of a file and returns it as a string.

    Args:
        file_name (str): The path to the file.

    Returns:
        str: The file contents.
    """
    with open(os.path.join(Path(os.path.dirname(__file__)), file_name), encoding='utf-8') as _file:
        return _file.read()


class StrategyResponse(BaseModel):
    """ # Strategy Response
    This class is a Pydantic model that represents the response of the strategy endpoint.

    Attributes:
        strategy (str): The strategy to play with the card in **early** or **late** game.
        status_code (int): The status code of the response. Default is 200.
    """
    strategy: str
    status_code: int = 200


app = FastAPI(
    title='Gods Unchained API',
    description="""
This is an API that receive the card ID of a card from the game Gods Unchained and return the strategy to play with it in **early** or **late** game.

## How to use the API
The API works with endpoints, each one with a specific task.

Basically, you need to send a request to the endpoint, and the API will return a response with the result of the task.

The API has the following endpoints:
- `/strategy`: This endpoint receives the card ID and returns the strategy to play.
- `/url-list`: This endpoint returns a list with all the endpoints of the API.

## Source code
The source code of this project can be found at [GitHub](https://github.com/Lucs1590/gods-unchained).
""",
    version=re.findall(
        re.compile(r'[0-9]+\.[0-9]+\.[0-9]+'),
        read('__version__.py')
    )[0]
)

app.mount('/prometheus', WSGIMiddleware(make_wsgi_app()))

security = HTTPBasic(
    description='This is a basic authentication to access the API endpoints.'
)
request_time, request_count, cache_count, normal_request_count = (
    Summary('request_processing_seconds', 'Time spent processing request'),
    Counter('request_count', 'Total request count'),
    Counter('cache_count', 'Total requests got from cache'),
    Counter('normal_request_count', 'Total requests got from the normal process')
)
allowed_users = {
    'leonardosilva': 'silvaleonardo',
    'samuel.silva': 'silva.samuel',
    'cunha.rodrigo': 'rodrigo.cunha',
    'lucas.brito': 'brito.lucas'
}

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.config_set('maxmemory-policy', 'allkeys-lru')
except redis.exceptions.ConnectionError:
    redis_client = None

try:
    dataframe = pd.read_csv('data/cards.csv')
except FileNotFoundError:
    dataframe = pd.DataFrame()
    print('Reference file not found.')


@app.get('/', include_in_schema=False)
def root():
    """ # Root Path
    This function redirects the user to the documentation page.

    Returns:
        RedirectResponse: A redirect response to the documentation page.
    """
    return RedirectResponse('/docs')


@app.get('/strategy', tags=['Strategy'], response_model=StrategyResponse)
async def retrieve_card_strategy(
    credentials: HTTPBasicCredentials = Depends(security),
    card_id: str = Query(
        ...,
        title='Card ID',
        description='The card ID of a card from the game Gods Unchained.',
        regex=r'^\d+$'
    )
) -> dict:
    """ # Strategy
    This endpoint receives the card ID of a card from the game Gods Unchained and return the strategy to play with it in **early** or **late** game.

    Returns:
        dict: The strategy to play with the card in **early** or **late** game.
    """
    strategy = None
    request_count.inc()

    with request_time.time():
        if credentials.username not in allowed_users or allowed_users[credentials.username] != credentials.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )

        if redis_client:
            strategy = redis_client.get(card_id)
            if strategy:
                cache_count.inc()
                return StrategyResponse(strategy=strategy.decode())

        card = dataframe[dataframe['id'] == int(card_id)]
        if card.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Card ID not found."
            )

        strategy = card['strategy'].values[0]

        if redis_client:
            redis_client.set(card_id, strategy)
            redis_client.expire(card_id, 60 * 15)  # 15 minutes

        normal_request_count.inc()
        return StrategyResponse(strategy=strategy)


@app.get("/url-list", include_in_schema=False, response_model=List[dict])
def get_all_urls():
    """ # URL List
    This endpoint returns a list with all the endpoints of the API.

    Returns:
        List[dict]: A list with all the endpoints of the API.
    """
    urls = [{
        "path": route.path,
        "name": route.name,
        'methods': route.methods
    } for route in app.routes]
    return urls
