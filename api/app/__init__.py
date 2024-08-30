from flask import Flask, jsonify
from api.app.bp.api_v1 import v1_bp


def create_app(llm):
    app = Flask(__name__)
    if not llm.vllm:
        llm.eval_load_model()
    app.config["llm"] = llm

    app.register_blueprint(v1_bp, url_prefix='/v1')

    @app.route('/')
    def index():
        return jsonify({"msg": "SuperAdapters API"})

    return app
