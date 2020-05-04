import numpy as np
import numba
critical_mass = 4


@numba.jit(nopython=True)
def get_unstable_cells(board):
    unstable_cells = []
    height, width = board.shape
    for i in range(width):
        for j in range(height):
            if (((i == 0 and j == 0) or
                    (i == width-1 and j == 0) or
                    (i == width-1 and j == height-1) or
                    (i == 0 and j == height-1))
                    and np.abs(board[j][i]) == critical_mass - 2):
                unstable_cells.append((i, j))
            elif ((i == 0 or
                    i == width-1 or
                    j == 0 or
                    j == height-1)
                    and np.abs(board[j][i]) == critical_mass - 1):
                unstable_cells.append((i, j))
            elif np.abs(board[j][i]) >= critical_mass:
                unstable_cells.append((i, j))
    return unstable_cells


@numba.jit(nopython=True)
def get_total_degrees_by_color(board):
    return {-1: (np.abs(board).sum() - board.sum())//2, 1: (np.abs(board).sum() + board.sum())//2}


@numba.jit(nopython=True)
def explode(board, cell):
    x, y = cell
    color = np.sign(board[y][x])
    height, width = board.shape
    frame = np.zeros(board.shape, np.int8)

    frame[y][x] = -board[y][x]
    if y - 1 >= 0:
        frame[y - 1][x] = color if np.sign(board[y - 1][x]
                                           ) == color else color*(1+2*np.abs((board[y - 1][x])))
    if y + 1 < height:
        frame[y + 1][x] = color if np.sign(board[y + 1][x]
                                           ) == color else color*(1+2*np.abs((board[y + 1][x])))
    if x - 1 >= 0:
        frame[y][x-1] = color if np.sign(board[y][x-1]
                                         ) == color else color*(1+2*np.abs((board[y][x-1])))
    if x + 1 < width:
        frame[y][x + 1] = color if np.sign(board[y][x+1]
                                           ) == color else color*(1+2*np.abs((board[y][x+1])))
    return frame+board


@numba.jit(nopython=True)
def get_legal_cell_locs(board, color):
    if color == -1:
        return [(ix, iy) for ix, iy in np.ndindex(board.shape) if board[ix, iy] <= 0]
    else:
        return [(ix, iy) for ix, iy in np.ndindex(board.shape) if board[ix, iy] >= 0]

@numba.jit(nopython=True)
def is_gameover(board):
    t_deg = get_total_degrees_by_color(board)
    return (t_deg[1] == 0 or t_deg[-1] == 0) and (t_deg[1] + t_deg[-1]) > 2


@numba.jit(nopython=True)
def play(board, x, y, color):
    if is_gameover(board):
        return board
    frame = np.zeros(board.shape, np.int8)
    frame[x][y] = color
    board = board + frame
    unstable_cells = get_unstable_cells(board)
    while unstable_cells:
        for cell in unstable_cells:
            board = explode(board, cell)
        unstable_cells = get_unstable_cells(board)
        if is_gameover(board):
            return board
    return board
