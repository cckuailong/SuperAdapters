import json
from flask import request, jsonify, current_app, abort
from flask.blueprints import Blueprint
from api.utils.auth import check_token

v1_bp = Blueprint("v1", __name__)


@v1_bp.errorhandler(500)
def handle_error(req):
    return jsonify({'success': False, 'msg': "Server busy. try later"})


@v1_bp.before_request
def token_and_query_check():
    token = request.args.get("token")
    if not token:
        abort(jsonify({'succeed': False, 'msg': 'Missing Param: token'}))
    dataj = request.get_json()
    if "user" not in dataj:
        abort(jsonify({'succeed': False, 'msg': 'Missing Param: user'}))
    user = dataj["user"]

    if not check_token(user, token, dataj):
        abort(jsonify({'succeed': False, 'msg': 'Invalid token'}))

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
    llm = current_app.config["llm"]
    resp = llm.evaluate(instruction, input)

    return json.dumps({'succeed': True, 'text': resp}, ensure_ascii=False)
