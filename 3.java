import java.util.*;

public class AStar8Puzzle {

    // Print the state
    static void printState(int[][] state) {
        for (int[] row : state) {
            for (int val : row) {
                System.out.print(val + " ");
            }
            System.out.println();
        }
    }

    // Find blank (0)
    static int[] findBlank(int[][] state) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] == 0) {
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }

    static int[][] copyState(int[][] state) {
        int[][] newState = new int[3][3];
        for (int i = 0; i < 3; i++) {
            newState[i] = state[i].clone();
        }
        return newState;
    }

    static int[][] moveUp(int[][] state) {
        int[] pos = findBlank(state);
        int i = pos[0], j = pos[1];
        if (i > 0) {
            int[][] newState = copyState(state);
            newState[i][j] = newState[i - 1][j];
            newState[i - 1][j] = 0;
            return newState;
        }
        return null;
    }

    static int[][] moveDown(int[][] state) {
        int[] pos = findBlank(state);
        int i = pos[0], j = pos[1];
        if (i < 2) {
            int[][] newState = copyState(state);
            newState[i][j] = newState[i + 1][j];
            newState[i + 1][j] = 0;
            return newState;
        }
        return null;
    }

    static int[][] moveLeft(int[][] state) {
        int[] pos = findBlank(state);
        int i = pos[0], j = pos[1];
        if (j > 0) {
            int[][] newState = copyState(state);
            newState[i][j] = newState[i][j - 1];
            newState[i][j - 1] = 0;
            return newState;
        }
        return null;
    }

    static int[][] moveRight(int[][] state) {
        int[] pos = findBlank(state);
        int i = pos[0], j = pos[1];
        if (j < 2) {
            int[][] newState = copyState(state);
            newState[i][j] = newState[i][j + 1];
            newState[i][j + 1] = 0;
            return newState;
        }
        return null;
    }

    // Heuristic: number of misplaced tiles
    static int calculateHeuristic(int[][] state, int[][] goal) {
        int h = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] != goal[i][j]) {
                    h++;
                }
            }
        }
        return h;
    }

    static String stateToString(int[][] state) {
        StringBuilder sb = new StringBuilder();
        for (int[] row : state) {
            for (int val : row) {
                sb.append(val);
            }
        }
        return sb.toString();
    }

    static void aStar(int[][] initialState, int[][] goalState) {

        // OPEN: (f, g, state)
        List<Node> OPEN = new ArrayList<>();
        Set<String> CLOSED = new HashSet<>();

        int h0 = calculateHeuristic(initialState, goalState);
        OPEN.add(new Node(h0, 0, initialState));

        while (!OPEN.isEmpty()) {

            // Get node with minimum f
            Node current = Collections.min(OPEN, Comparator.comparingInt(n -> n.f));
            OPEN.remove(current);

            CLOSED.add(stateToString(current.state));

            printState(current.state);
            System.out.println();

            if (Arrays.deepEquals(current.state, goalState)) {
                System.out.println("Solution found!");
                return;
            }

            int[][][] moves = {
                    moveUp(current.state),
                    moveDown(current.state),
                    moveLeft(current.state),
                    moveRight(current.state)
            };

            for (int[][] successor : moves) {
                if (successor == null) continue;

                String key = stateToString(successor);
                if (CLOSED.contains(key)) continue;

                int gNew = current.g + 1;
                int hNew = calculateHeuristic(successor, goalState);
                int fNew = gNew + hNew;

                OPEN.add(new Node(fNew, gNew, successor));
            }
        }

        System.out.println("No solution found.");
    }

    // Node class
    static class Node {
        int f, g;
        int[][] state;

        Node(int f, int g, int[][] state) {
            this.f = f;
            this.g = g;
            this.state = state;
        }
    }

    // Main method
    public static void main(String[] args) {

        int[][] initialState = {
                {1, 2, 3},
                {8, 0, 4},
                {7, 6, 5}
        };

        int[][] goalState = {
                {2, 8, 1},
                {0, 4, 3},
                {7, 6, 5}
        };

        aStar(initialState, goalState);
    }
}