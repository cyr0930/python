from flask import Flask
from flask import db
from flask.controller import article_api, comment_api

app = Flask(__name__)


if __name__ == "__main__":
    db.init_db()
    app.register_blueprint(article_api.api)
    app.register_blueprint(comment_api.api)
    app.run(host="0.0.0.0")
