from datetime import datetime
from app import db
from app.service import article_service, comment_service


def test():
    db.init_db()

    article_json = {
        'title': '1st Article',
        'body': 'Hello world',
        'tags': 'small talk',
        'published_from': datetime.now(),
    }
    article_id = article_service.add(article_json)

    comment_json = {
        'author': 'Zack',
        'content': 'like',
        'article_id': article_id,
    }
    comment_id = comment_service.add(comment_json)

    article = article_service.get(article_id)
    del article['lines']
    assert article_json == article
    comment = comment_service.get(comment_id)
    del comment['created_at']
    assert comment_json == comment


test()
