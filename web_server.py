import json
from flask import Flask, jsonify, request,send_file
from flask_restful import Resource, Api

import modules.globals
from modules import core
from threading import Thread
import time

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/')


@app.route('/api')
def hello_world():
    return 'Hello World! 你好'


def thread_func(*args):
    executionId = ""
    # for e in args:
    #     executionId = e
    core.start(execution_id="".join(args))
    # core.start(execution_id=args[0])


@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.get_data(as_text=True)
    jsonObj = json.loads(data)
    print("request data:", jsonObj)
    core.set_params(jsonObj)
    executionId = int(time.time()*1000)
    t = Thread(target=thread_func, args=(str(executionId)))
    # t = Thread(target=start)
    t.start()
    return {"code": 200, "msg": "success", "data": {"execution_id": executionId,"output":modules.globals.output_path}}


@app.route('/api/status/get', methods=['GET'])
def get_status():
    executionId = request.args.get('executionId', None)
    return core.getExecutionStatus(executionId)\


@app.route('/api/download', methods=['GET'])
def download_file():
    filePath = request.args.get('filePath', None)
    return send_file(filePath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000)
