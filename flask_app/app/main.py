import os, logging
from flask import Flask
from elasticapm.contrib.flask import ElasticAPM
from elasticapm.handlers.logging import LoggingHandler
from app import db
from app.controller import article_api, comment_api

app = Flask(__name__)
app.config['ELASTIC_APM'] = {
    'SERVICE_NAME': 'flask_service',
    'SERVER_URL': f'http://{os.getenv("ELASTIC_HOST")}:8200'
}
apm = ElasticAPM(app)

db.init_db()
app.register_blueprint(article_api.api)
app.register_blueprint(comment_api.api)

handler = LoggingHandler(client=apm.client)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

