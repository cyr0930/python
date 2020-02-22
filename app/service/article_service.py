from app.documents import Article


def get(article_id):
    try:
        article = Article.get(article_id)
        return article.to_dict()
    except Exception as e:
        print(e)
        return None


def add(data):
    article = Article(title=data['title'], body=data['body'], tags=data['tags'], published_from=data['published_from'])
    article.save()
    return article.meta.id
