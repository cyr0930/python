from elasticsearch_dsl.connections import connections
from app.documents import Article, Comment


def init_db():
    connections.create_connection(hosts=['192.168.35.75'])
    Article.init()
    Comment.init()
