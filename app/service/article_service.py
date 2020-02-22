from datetime import datetime
from app.documents import Article
from app.service import comment_service
import threading

datetime_format = '%Y-%m-%d %H:%M:%S'


class AsyncComment(threading.Thread):
    def __init__(self, comment):
        threading.Thread.__init__(self)
        self.comment = comment

    def run(self):
        print('run add_comment job in background')
        comment_service.add(self.comment)


def get(article_id):
    try:
        article = Article.get(article_id)
        return article.to_dict()
    except Exception as e:
        print(e)
        return None


def _add(data):
    article = Article(title=data['title'], body=data['body'], tags=data['tags'],
                      published_from=datetime.strptime(data['published_from'], datetime_format))
    article.save()
    return article.meta.id


def add(data):
    if 'comments' in data:
        comments = data.pop('comments')
        article_id = _add(data)
        for comment in comments:
            comment['article_id'] = article_id
            AsyncComment(comment).start()
        return article_id
    else:
        return _add(data)
