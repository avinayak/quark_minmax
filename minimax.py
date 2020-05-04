import numpy as np
import quark


def get_best_move(board, legal_moves):
    best_move = None
    best_score = -9999999999999999999
    for move in legal_moves:
        copy_board = np.copy(board)
        copy_board = quark.play(copy_board, move[0], move[1], -1)
        score = quark.get_total_degrees_by_color(copy_board)[-1]
        print(move, score)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def generate_child_nodes(board, maximizingPlayer):
    child_nodes = []
    player = -1 if maximizingPlayer else 1
    legal_moves = quark.get_legal_cell_locs(board, player)
    for move in legal_moves:
        copy_board = np.copy(board)
        copy_board = quark.play(copy_board, move[0], move[1], player)
        child_nodes.append({"board": copy_board, "move": [move]})
    return child_nodes


def heuristicValue(board):
    return quark.get_total_degrees_by_color(board)[-1]/(quark.get_total_degrees_by_color(board)[1]+1)


def minimax(node, depth, a, b,maximizingPlayer):
    if depth == 0 or quark.is_gameover(node['board']):
        return {'value': heuristicValue(node['board']), 'move': node['move']}
    if maximizingPlayer:
        max_value = -100000000000000
        move = None
        for child_node in generate_child_nodes(node['board'], maximizingPlayer):
            value = minimax(child_node, depth - 1, a, b, False)
            if value['value'] > max_value:
                max_value = value['value']
                move = value['move']
            a = max(a, max_value)
            if a>b:
                break
        #print({'value': max_value, 'move': node['move']+move}, depth, "max")
        return {'value': max_value, 'move': node['move']+move}
    else:
        min_value = 100000000000000
        move = None
        for child_node in generate_child_nodes(node['board'], maximizingPlayer):
            value = minimax(child_node, depth - 1, a, b, True)
            if value['value'] < min_value:
                min_value = value['value']
                move = value['move']
            b = min(b, min_value)
            if a >= b:
                break
        #print({'value': min_value, 'move': node['move']+move}, depth, "min")
        return {'value': min_value, 'move': node['move']+move}
