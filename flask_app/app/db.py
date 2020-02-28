import os
from elasticsearch_dsl.connections import connections
from app.documents import Article, Comment


def init_db():
    connections.create_connection(hosts=[os.getenv('ELASTIC_HOST')])
    Article.init()
    Comment.init()
