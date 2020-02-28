from elasticsearch_dsl.connections import connections
from flask_app.documents import Article, Comment


def init_db():
    connections.create_connection(hosts=['localhost'])
    Article.init()
    Comment.init()
