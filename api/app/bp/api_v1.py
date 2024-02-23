import json
from flask import request, jsonify, current_app
from flask.blueprints import Blueprint

v1_bp = Blueprint("v1", __name__)


@v1_bp.errorhandler(500)
def handle_error(req):
    return jsonify({'success': False, 'msg': "Server busy. try later"})


@v1_bp.route('/call', methods=['POST'])
def call():
    dataj = request.get_json()
    if "instruction" not in dataj:
        instruction = ""
    else:
        instruction = dataj["instruction"]
    if "input" not in dataj:
        return jsonify({'succeed': False, 'msg': 'Missing Param: input'})

    input = dataj["input"]
    model = current_app.config["model"]
    llm = current_app.config["llm"]
    resp = llm.evaluate(model, instruction, input)

    return json.dumps({'succeed': True, 'text': resp}, ensure_ascii=False)
