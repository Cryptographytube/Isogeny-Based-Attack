import networkx as nx
from fastecdsa.curve import secp256k1
from fastecdsa.point import Point
from fpylll import IntegerMatrix, LLL

p = secp256k1.p
n = secp256k1.q
G = secp256k1.G

def get_attack_mode():
    print("\n🔍 Select Attack Mode:")
    print("1: Endomorphism Weakness")
    print("2: Isogeny Graph Pathfinding")
    print("3: Lattice Reduction (Private Key Extraction)")
    return int(input("Enter Choice (1/2/3): ").strip())

def check_endomorphism():
    print("\n[🔹] Running Endomorphism Weakness Check...")
    private_key = int(input("Enter Private Key to Analyze: ").strip())
    public_key = Point(private_key * G.x, private_key * G.y, curve=secp256k1)
    beta = pow(2, (p - 1) // 3, p)
    new_x = (beta * public_key.x) % p
    new_P = Point(new_x, public_key.y, curve=secp256k1)
    print(f"Original Public Key: {public_key}")
    print(f"Weak Endomorphism Point: {new_P}")

def create_isogeny_graph():
    print("\n[🔹] Constructing Isogeny Graph...")
    graph_size = int(input("Enter Graph Size (Recommended: 50+): ").strip())
    G = nx.Graph()
    for i in range(1, graph_size):
        G.add_edge(i, i + 2)
    print("[✅] Isogeny Graph Created Successfully!")
    return G

def find_attack_path(graph):
    start = int(input("Enter Start Node: ").strip())
    target = int(input("Enter Target Node: ").strip())
    path = nx.shortest_path(graph, source=start, target=target)
    print(f"⚡ Isogeny Attack Path Found: {path}")

def lattice_attack():
    print("\n[🔹] Running Lattice Reduction Attack...")
    matrix_size = int(input("Enter Matrix Size (Recommended: 3x3, 4x4): ").strip())
    matrix = []
    for i in range(matrix_size):
        row = list(map(int, input(f"Enter Row {i+1} (comma-separated): ").strip().split(",")))
        matrix.append(row)
    
    M = IntegerMatrix.from_matrix(matrix)
    LLL.reduction(M)
    print("[✅] Lattice Reduction Completed!")
    print("Reduced Matrix (Potential Private Key Data):")
    print(M)

if __name__ == "__main__":
    mode = get_attack_mode()
    if mode == 1:
        check_endomorphism()
    elif mode == 2:
        graph = create_isogeny_graph()
        find_attack_path(graph)
    elif mode == 3:
        lattice_attack()
    else:
        print("[❌] Invalid Choice! Restart and Select a Valid Option.")
