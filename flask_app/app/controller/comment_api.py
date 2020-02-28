from flask import Blueprint, jsonify, request
from werkzeug.exceptions import abort
from app.service import comment_service

api = Blueprint('comment', __name__, url_prefix='/api/comments')


@api.route('/<string:comment_id>')
def get(comment_id):
    comment = comment_service.get(comment_id)
    if not comment:
        abort(404, f"Article {comment_id} doesn't exist.")
    return jsonify(comment)


@api.route('/', methods=['POST'])
def add():
    return jsonify(comment_service.add(request.get_json()))
