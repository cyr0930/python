from flask import current_app
from app.documents import Article
from app.service import comment_service
import threading


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
        current_app.logger.error(f"Article {article_id} doesn't exist.")
        return None


def get_by_tag(tag):
    s = Article.search().filter('match', tags=tag)[:1000]
    articles = s.execute()
    return articles.to_dict()


def _add(data):
    article = Article.from_json(data)
    article.save(refresh='wait_for')
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
