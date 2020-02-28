from app.documents import Comment


def get(comment_id):
    try:
        comment = Comment.get(comment_id)
        return comment.to_dict()
    except Exception as e:
        print(e)
        return None


def add(data):
    comment = Comment.from_json(data)
    comment.save(refresh='wait_for')
    return comment.meta.id
