
      
def print_state(state):
    for row in state:
        print(" ".join(map(str, row)))


def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j


def move_up(state):
    i, j = find_blank(state)
    if i > 0:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        return new_state
    return None


def move_down(state):
    i, j = find_blank(state)
    if i < 2:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
        return new_state
    return None


def move_left(state):
    i, j = find_blank(state)
    if j > 0:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        return new_state
    return None


def move_right(state):
    i, j = find_blank(state)
    if j < 2:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        return new_state
    return None


def calculate_heuristic(state, goal_state):
    h = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                h += 1
    return h


def a_star(initial_state, goal_state):
    OPEN = []
    CLOSED = set()

    h0 = calculate_heuristic(initial_state, goal_state)
    OPEN.append((h0, 0, initial_state))  # (f, g, state)

    while OPEN:
        # pick node with minimum f
        f, g, current_state = min(OPEN, key=lambda x: x[0])
        OPEN.remove((f, g, current_state))

        state_key = tuple(map(tuple, current_state))
        CLOSED.add(state_key)

        print_state(current_state)
        print()

        if current_state == goal_state:
            print("Solution found!")
            return

        successors = [
            move_up(current_state),
            move_down(current_state),
            move_left(current_state),
            move_right(current_state)
        ]

        for successor in successors:
            if successor is None:
                continue

            succ_key = tuple(map(tuple, successor))
            if succ_key in CLOSED:
                continue

            g_succ = g + 1
            h_succ = calculate_heuristic(successor, goal_state)
            f_succ = g_succ + h_succ

            # check if better path exists in OPEN
            in_open = False
            for f_old, g_old, state_old in OPEN:
                if state_old == successor and g_old <= g_succ:
                    in_open = True
                    break

            if not in_open:
                OPEN.append((f_succ, g_succ, successor))

    print("No solution found.")