import os
import re

from typing import List
from pathlib import Path

import redis

from fastapi import Depends, FastAPI, Query, HTTPException, status
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse

from prometheus_client import make_wsgi_app


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


@app.get('/', include_in_schema=False)
def root():
    """ # Root Path
    This function redirects the user to the documentation page.

    Returns:
        RedirectResponse: A redirect response to the documentation page.
    """
    return RedirectResponse('/docs')


@app.get('/strategy', tags=['Strategy'], response_model=dict)
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
    if credentials.username not in allowed_users or allowed_users[credentials.username] != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    if redis_client:
        strategy = redis_client.get(card_id)
        if strategy:
            return {'strategy': strategy.decode()}

    if redis_client:
        redis_client.set(card_id, strategy)
        redis_client.expire(card_id, 60 * 15)  # 15 minutes

    return {'data': strategy}


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
