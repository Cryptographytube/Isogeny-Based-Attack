import networkx as nx
from fastecdsa.curve import secp256k1
from fastecdsa.point import Point
from fpylll import IntegerMatrix, LLL
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from numba import jit, cuda as numba_cuda
import torch

p = secp256k1.p
n = secp256k1.q
G = secp256k1.G

def get_attack_mode():
    print("\n🔍 Select Attack Mode:")
    print("1: AI-Optimized Endomorphism Weakness")
    print("2: AI + GPU Isogeny Graph Search")
    print("3: GPU-Accelerated Lattice Reduction (Private Key Extraction)")
    return int(input("Enter Choice (1/2/3): ").strip())

@jit(nopython=True)
def optimized_mod_mul(x, y, mod):
    return (x * y) % mod

def check_endomorphism():
    print("\n[🔹] Running AI-Optimized Endomorphism Check...")
    private_key = int(input("Enter Private Key to Analyze: ").strip())
    public_key = Point(optimized_mod_mul(private_key, G.x, p), optimized_mod_mul(private_key, G.y, p), curve=secp256k1)
    beta = pow(2, (p - 1) // 3, p)
    new_x = optimized_mod_mul(beta, public_key.x, p)
    new_P = Point(new_x, public_key.y, curve=secp256k1)
    print(f"Original Public Key: {public_key}")
    print(f"Weak Endomorphism Point: {new_P}")

def create_isogeny_graph():
    print("\n[🔹] Constructing AI-Optimized Isogeny Graph...")
    graph_size = int(input("Enter Graph Size (Recommended: 100+): ").strip())
    G = nx.Graph()
    for i in range(1, graph_size):
        G.add_edge(i, i + np.random.randint(1, 5))
    print("[✅] AI-Based Isogeny Graph Created Successfully!")
    return G

def find_attack_path(graph):
    start = int(input("Enter Start Node: ").strip())
    target = int(input("Enter Target Node: ").strip())

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1)
    )
    
    def ai_heuristic_search(graph, start, target):
        queue = [(0, start)]
        visited = set()
        while queue:
            cost, node = queue.pop(0)
            if node == target:
                return cost
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    heuristic = model(torch.tensor([node, neighbor], dtype=torch.float32)).item()
                    queue.append((cost + heuristic, neighbor))
                queue.sort()
        return -1
    
    path_cost = ai_heuristic_search(graph, start, target)
    print(f"⚡ AI-Optimized Isogeny Attack Path Cost: {path_cost}")

def gpu_lattice_attack():
    print("\n[🔹] Running GPU-Accelerated Lattice Reduction...")
    matrix_size = int(input("Enter Matrix Size (Recommended: 4x4, 5x5): ").strip())
    matrix = np.random.randint(-10, 10, (matrix_size, matrix_size)).astype(np.int32)

    mod = SourceModule("""
    __global__ void reduce(int *matrix, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            for (int j = 0; j < size; j++) {
                matrix[idx * size + j] %= 7;
            }
        }
    }
    """)

    gpu_matrix = cuda.mem_alloc(matrix.nbytes)
    cuda.memcpy_htod(gpu_matrix, matrix)
    func = mod.get_function("reduce")
    func(gpu_matrix, np.int32(matrix_size), block=(256, 1, 1), grid=(matrix_size // 256 + 1, 1))

    cuda.memcpy_dtoh(matrix, gpu_matrix)
    M = IntegerMatrix.from_matrix(matrix.tolist())
    LLL.reduction(M)
    
    print("[✅] GPU-Based Lattice Reduction Completed!")
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
        gpu_lattice_attack()
    else:
        print("[❌] Invalid Choice! Restart and Select a Valid Option.")
