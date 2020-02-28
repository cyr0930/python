from datetime import datetime
from app import db, documents
from app.service import article_service, comment_service


def test():
    article_json = {
        'title': '1st Article',
        'body': 'Hello world',
        'tags': 'small talk',
        'published_from': '2020-02-23 00:10:00',
    }
    len1 = len(article_service.get_by_tag('small talk')['hits']['hits'])
    article_id = article_service.add(article_json)
    len2 = len(article_service.get_by_tag('small talk')['hits']['hits'])
    # test sync
    assert len1+1 == len2

    comment_json = {
        'author': 'Zack',
        'content': 'like',
        'article_id': article_id,
    }
    comment_id = comment_service.add(comment_json)

    article = article_service.get(article_id)
    del article['lines']
    article_json['published_from'] = datetime.strptime(article_json['published_from'], documents.datetime_format)
    assert article_json == article
    comment = comment_service.get(comment_id)
    del comment['created_at']
    assert comment_json == comment


def test_async():
    article_json = {
        'title': 'Async Article',
        'body': 'Hello world',
        'tags': 'small talk',
        'published_from': '2020-02-23 00:10:00',
        'comments': [
            {'author': 'Zack', 'content': 'like'},
            {'author': 'Sheldon', 'content': 'dislike'},
        ],
    }
    article_id = article_service.add(article_json)

    article = article_service.get(article_id)
    del article['lines']
    article_json['published_from'] = datetime.strptime(article_json['published_from'], documents.datetime_format)
    assert article_json == article


if __name__ == "__main__":
    db.init_db()
    test()
    test_async()
