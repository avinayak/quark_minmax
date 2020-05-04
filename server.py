from flask import Flask, request, jsonify
from flask_cors import CORS
import quark
import random
import numpy as np
import time
import rle
from atom import Atom
import ai

app = Flask(__name__, static_folder='web', root_path='.')
CORS(app)

movecache = {}


@app.route('/')
def root():
    return app.send_static_file('./index.html')


initialState = Atom()
@app.route('/api/move/', methods=['GET', 'POST'])
def move():
    print(request)
    start_time = time.time()
    board = -np.asarray(request.json, np.int8)
    encboard = rle.encode(board)
    if encboard in movecache:
        action = movecache[encboard]
        return jsonify({'x': int(action.y), 'y': int(action.x)})
    print(board)

    val = ai.minimax({"board": board, 'move': []}, 3, -1e100, 1e100, True)
    print(val)
    y, x = val['move'][0]
    end_time = time.time()
    print("processed {} board in {}".format(board.shape,
                                            end_time - start_time))
    return jsonify({'x': x, 'y': y})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3000')
