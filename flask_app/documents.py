from datetime import datetime
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text

datetime_format = '%Y-%m-%d %H:%M:%S'


class Article(Document):
    title = Text(fields={'keyword': Keyword()})
    body = Text()
    tags = Keyword()
    published_from = Date()
    lines = Integer()

    class Index:
        name = 'blog-articles'

    def save(self, **kwargs):
        self.lines = len(self.body.split())
        return super(Article, self).save(**kwargs)

    @staticmethod
    def from_json(data):
        return Article(title=data['title'], body=data['body'], tags=data['tags'],
                       published_from=datetime.strptime(data['published_from'], datetime_format))


class Comment(Document):
    author = Text(fields={'keyword': Keyword()})
    content = Text()
    created_at = Date()
    article_id = Keyword()

    class Index:
        name = 'blog-comments'

    def save(self, **kwargs):
        self.created_at = datetime.now()
        return super(Comment, self).save(**kwargs)

    @staticmethod
    def from_json(data):
        return Comment(author=data['author'], content=data['content'], article_id=data['article_id'])
