from flask import Blueprint, jsonify, request
from werkzeug.exceptions import abort
from app.service import article_service

api = Blueprint('article', __name__, url_prefix='/api/articles')


@api.before_request
def before_article_api():
    print('request article api')


@api.route('/<string:article_id>')
def get(article_id):
    article = article_service.get(article_id)
    if not article:
        abort(404, f"Article {article_id} doesn't exist.")
    return jsonify(article)


@api.route('/', methods=['POST'])
def add():
    return jsonify(article_service.add(request.get_json()))
