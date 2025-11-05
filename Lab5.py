import heapq
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict

class Graph:
    """Класс для представления графа"""
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.adj_matrix = None
        self.num_vertices = 0
    
    def load_from_adjacency_matrix(self, filename: str):
        """Загрузка графа из матрицы смежности"""
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            # Читаем матрицу смежности
            matrix = []
            for line in lines:
                row = [float(x) if x != 'inf' else float('inf') for x in line.strip().split()]
                matrix.append(row)
            
            self.adj_matrix = np.array(matrix)
            self.num_vertices = len(self.adj_matrix)
            self.vertices = list(range(self.num_vertices))
            
            # Преобразуем матрицу смежности в список ребер
            self.edges = []
            for i in range(self.num_vertices):
                for j in range(i + 1, self.num_vertices):
                    if self.adj_matrix[i][j] != float('inf'):
                        self.edges.append((i, j, self.adj_matrix[i][j]))
            
            print(f"Граф загружен: {self.num_vertices} вершин, {len(self.edges)} ребер")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False
    
    def load_from_edges(self, filename: str):
        """Загрузка графа из списка ребер"""
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            edges = []
            vertices_set = set()
            
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        u, v, weight = int(parts[0]), int(parts[1]), float(parts[2])
                        edges.append((u, v, weight))
                        vertices_set.add(u)
                        vertices_set.add(v)
            
            self.edges = edges
            self.vertices = sorted(vertices_set)
            self.num_vertices = len(self.vertices)
            
            # Создаем матрицу смежности
            self.adj_matrix = np.full((self.num_vertices, self.num_vertices), float('inf'))
            for u, v, weight in edges:
                self.adj_matrix[u][v] = weight
                self.adj_matrix[v][u] = weight
            
            print(f"Граф загружен: {self.num_vertices} вершин, {len(self.edges)} ребер")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False

class UnionFind:
    """Структура данных для объединения непересекающихся множеств"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Находит корень множества для элемента x"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Объединяет множества, содержащие x и y"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Уже в одном множестве
        
        # Объединение по рангу
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

class KruskalAlgorithm:
    """Реализация алгоритма Краскала"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.mst_edges = []
        self.total_weight = 0
    
    def find_mst(self) -> Tuple[List[Tuple[int, int, float]], float]:
        """Поиск минимального остовного дерева"""
        start_time = time.time()
        
        # Сортируем ребра по весу
        sorted_edges = sorted(self.graph.edges, key=lambda x: x[2])
        
        uf = UnionFind(self.graph.num_vertices)
        self.mst_edges = []
        self.total_weight = 0
        
        for u, v, weight in sorted_edges:
            if uf.union(u, v):
                self.mst_edges.append((u, v, weight))
                self.total_weight += weight
                
                # Если построили MST с n-1 ребром, выходим
                if len(self.mst_edges) == self.graph.num_vertices - 1:
                    break
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        return self.mst_edges, self.total_weight
    
    def get_execution_time(self) -> float:
        """Возвращает время выполнения алгоритма"""
        return getattr(self, 'execution_time', 0)

class GraphVisualizer:
    """Класс для визуализации графа"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def draw_graph(self, graph: Graph, mst_edges: List[Tuple[int, int, float]] = None, 
                   title: str = "Граф"):
        """Отрисовка графа с выделением MST"""
        G = nx.Graph()
        
        # Добавляем вершины
        for i in range(graph.num_vertices):
            G.add_node(i)
        
        # Добавляем ребра
        for u, v, weight in graph.edges:
            G.add_edge(u, v, weight=weight)
        
        # Создаем layout для вершин
        pos = nx.spring_layout(G, seed=42)
        
        # Создаем график
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Рисуем все ребра серым цветом
        nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color='gray', 
                              width=1, alpha=0.5)
        
        # Рисуем MST ребра красным цветом
        if mst_edges:
            mst_G = nx.Graph()
            for u, v, weight in mst_edges:
                mst_G.add_edge(u, v, weight=weight)
            nx.draw_networkx_edges(mst_G, pos, ax=self.ax, edge_color='red', 
                                  width=3, alpha=0.8)
        
        # Рисуем вершины
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color='lightblue', 
                              node_size=500, alpha=0.9)
        
        # Подписываем вершины
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_size=12, 
                               font_weight='bold')
        
        # Подписываем ребра весами
        edge_labels = {(u, v): f'{weight:.1f}' for u, v, weight in graph.edges}
        nx.draw_networkx_edge_labels(G, pos, ax=self.ax, edge_labels=edge_labels, 
                                   font_size=8)
        
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def draw_complexity_chart(self, complexity_data: Dict):
        """Отрисовка графика временной сложности"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График зависимости от числа вершин
        vertices = list(complexity_data['by_vertices'].keys())
        times_vertices = list(complexity_data['by_vertices'].values())
        
        ax1.plot(vertices, times_vertices, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Количество вершин')
        ax1.set_ylabel('Время выполнения (секунды)')
        ax1.set_title('Зависимость времени от числа вершин')
        ax1.grid(True, alpha=0.3)
        
        # График зависимости от числа ребер
        edges = list(complexity_data['by_edges'].keys())
        times_edges = list(complexity_data['by_edges'].values())
        
        ax2.plot(edges, times_edges, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Количество ребер')
        ax2.set_ylabel('Время выполнения (секунды)')
        ax2.set_title('Зависимость времени от числа ребер')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class GraphApp:
    """Графический интерфейс для работы с алгоритмом Краскала"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритм Краскала - Минимальное остовное дерево")
        self.root.geometry("800x600")
        
        self.graph = Graph()
        self.kruskal = None
        self.visualizer = GraphVisualizer()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Алгоритм Краскала", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Фрейм загрузки графа
        load_frame = ttk.LabelFrame(main_frame, text="Загрузка графа", padding="10")
        load_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(load_frame, text="Загрузить матрицу смежности", 
                  command=self.load_adjacency_matrix).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(load_frame, text="Загрузить список ребер", 
                  command=self.load_edges_list).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(load_frame, text="Сгенерировать тестовый граф", 
                  command=self.generate_test_graph).grid(row=0, column=2)
        
        # Информация о графе
        self.info_label = ttk.Label(main_frame, text="Граф не загружен", 
                                   font=('Arial', 10))
        self.info_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Фрейм алгоритма
        algo_frame = ttk.LabelFrame(main_frame, text="Алгоритм Краскала", padding="10")
        algo_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(algo_frame, text="Найти минимальное остовное дерево", 
                  command=self.find_mst).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Визуализировать граф", 
                  command=self.visualize_graph).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Анализ сложности", 
                  command=self.analyze_complexity).grid(row=0, column=2)
        
        # Фрейм результатов
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        result_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Текстовое поле для вывода результатов
        self.result_text = tk.Text(result_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Настройка весов строк и столбцов для растягивания
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
    
    def load_adjacency_matrix(self):
        """Загрузка графа из матрицы смежности"""
        filename = filedialog.askopenfilename(
            title="Выберите файл с матрицей смежности",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            if self.graph.load_from_adjacency_matrix(filename):
                self.info_label.config(
                    text=f"Граф загружен: {self.graph.num_vertices} вершин, {len(self.graph.edges)} ребер"
                )
                self.kruskal = KruskalAlgorithm(self.graph)
                self.log_message(f"Граф загружен из файла: {filename}")
    
    def load_edges_list(self):
        """Загрузка графа из списка ребер"""
        filename = filedialog.askopenfilename(
            title="Выберите файл со списком ребер",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            if self.graph.load_from_edges(filename):
                self.info_label.config(
                    text=f"Граф загружен: {self.graph.num_vertices} вершин, {len(self.graph.edges)} ребер"
                )
                self.kruskal = KruskalAlgorithm(self.graph)
                self.log_message(f"Граф загружен из файла: {filename}")
    
    def generate_test_graph(self):
        """Генерация тестового графа"""
        # Создаем полный граф с 6 вершинами
        self.graph.num_vertices = 6
        self.graph.vertices = list(range(6))
        self.graph.edges = []
        
        # Добавляем случайные ребра с весами
        import random
        for i in range(6):
            for j in range(i + 1, 6):
                weight = random.uniform(1, 10)
                self.graph.edges.append((i, j, weight))
        
        # Создаем матрицу смежности
        self.graph.adj_matrix = np.full((6, 6), float('inf'))
        for u, v, weight in self.graph.edges:
            self.graph.adj_matrix[u][v] = weight
            self.graph.adj_matrix[v][u] = weight
        
        self.info_label.config(
            text=f"Тестовый граф: {self.graph.num_vertices} вершин, {len(self.graph.edges)} ребер"
        )
        self.kruskal = KruskalAlgorithm(self.graph)
        self.log_message("Сгенерирован тестовый граф (6 вершин, полный граф)")
    
    def find_mst(self):
        """Поиск минимального остовного дерева"""
        if not self.kruskal:
            messagebox.showerror("Ошибка", "Сначала загрузите граф")
            return
        
        try:
            mst_edges, total_weight = self.kruskal.find_mst()
            execution_time = self.kruskal.get_execution_time()
            
            self.log_message("\n" + "="*50)
            self.log_message("МИНИМАЛЬНОЕ ОСТОВНОЕ ДЕРЕВО (Алгоритм Краскала)")
            self.log_message("="*50)
            self.log_message(f"Общий вес MST: {total_weight:.2f}")
            self.log_message(f"Время выполнения: {execution_time:.6f} секунд")
            self.log_message("\nРебра MST:")
            
            for i, (u, v, weight) in enumerate(mst_edges, 1):
                self.log_message(f"  {i}. {u} -- {v} (вес: {weight:.2f})")
            
            # Визуализация результата
            self.visualizer.draw_graph(self.graph, mst_edges, 
                                     "Минимальное остовное дерево (Алгоритм Краскала)")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении алгоритма: {e}")
    
    def visualize_graph(self):
        """Визуализация графа"""
        if not self.graph or self.graph.num_vertices == 0:
            messagebox.showerror("Ошибка", "Сначала загрузите граф")
            return
        
        try:
            self.visualizer.draw_graph(self.graph, title="Исходный граф")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при визуализации: {e}")
    
    def analyze_complexity(self):
        """Анализ временной сложности алгоритма"""
        complexity_data = self._measure_complexity()
        
        # Вывод результатов в текстовое поле
        self.log_message("\n" + "="*50)
        self.log_message("АНАЛИЗ ВРЕМЕННОЙ СЛОЖНОСТИ")
        self.log_message("="*50)
        
        self.log_message("\nЗависимость от числа вершин (при V-1 ребрах):")
        self.log_message("Вершины | Время (сек)")
        self.log_message("-" * 20)
        for vertices, time_val in complexity_data['by_vertices'].items():
            self.log_message(f"{vertices:7} | {time_val:.6f}")
        
        self.log_message("\nЗависимость от числа ребер (при 10 вершинах):")
        self.log_message("Ребра   | Время (сек)")
        self.log_message("-" * 20)
        for edges, time_val in complexity_data['by_edges'].items():
            self.log_message(f"{edges:7} | {time_val:.6f}")
        
        # Визуализация графиков
        self.visualizer.draw_complexity_chart(complexity_data)
    
    def _measure_complexity(self) -> Dict:
        """Измерение временной сложности на различных графах"""
        complexity_data = {'by_vertices': {}, 'by_edges': {}}
        
        # Тестирование зависимости от числа вершин
        print("Измерение зависимости от числа вершин...")
        for n in [5, 10, 15, 20, 25]:
            test_graph = self._generate_test_graph_by_vertices(n)
            kruskal = KruskalAlgorithm(test_graph)
            mst_edges, total_weight = kruskal.find_mst()
            complexity_data['by_vertices'][n] = kruskal.get_execution_time()
        
        # Тестирование зависимости от числа ребер
        print("Измерение зависимости от числа ребер...")
        for e in [10, 20, 30, 40, 50]:
            test_graph = self._generate_test_graph_by_edges(10, e)  # 10 вершин
            kruskal = KruskalAlgorithm(test_graph)
            mst_edges, total_weight = kruskal.find_mst()
            complexity_data['by_edges'][e] = kruskal.get_execution_time()
        
        return complexity_data
    
    def _generate_test_graph_by_vertices(self, n: int) -> Graph:
        """Генерация тестового графа с n вершинами"""
        graph = Graph()
        graph.num_vertices = n
        graph.vertices = list(range(n))
        graph.edges = []
        
        import random
        # Создаем связный граф (дерево + дополнительные ребра)
        for i in range(1, n):
            graph.edges.append((0, i, random.uniform(1, 10)))
        
        # Добавляем случайные ребра
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:  # 30% вероятность добавления ребра
                    graph.edges.append((i, j, random.uniform(1, 10)))
        
        return graph
    
    def _generate_test_graph_by_edges(self, n: int, e: int) -> Graph:
        """Генерация тестового графа с n вершинами и e ребрами"""
        graph = Graph()
        graph.num_vertices = n
        graph.vertices = list(range(n))
        graph.edges = []
        
        import random
        # Сначала создаем связный граф (n-1 ребер)
        for i in range(1, n):
            graph.edges.append((0, i, random.uniform(1, 10)))
        
        # Добавляем оставшиеся ребра случайным образом
        while len(graph.edges) < e:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not any((u, v, w) in graph.edges or (v, u, w) in graph.edges for w in [0]):
                graph.edges.append((u, v, random.uniform(1, 10)))
        
        return graph
    
    def log_message(self, message: str):
        """Добавление сообщения в текстовое поле"""
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.root.update()

def create_sample_files():
    """Создание примерных файлов для тестирования"""
    
    # Пример матрицы смежности (5 вершин)
    adjacency_matrix = """0 2 inf 6 inf
2 0 3 8 5
inf 3 0 inf 7
6 8 inf 0 9
inf 5 7 9 0"""
    
    with open('sample_adjacency.txt', 'w') as f:
        f.write(adjacency_matrix)
    
    # Пример списка ребер
    edges_list = """0 1 2
0 3 6
1 2 3
1 3 8
1 4 5
2 4 7
3 4 9"""
    
    with open('sample_edges.txt', 'w') as f:
        f.write(edges_list)
    
    print("Созданы примерные файлы: sample_adjacency.txt, sample_edges.txt")

def main():
    # Создаем примерные файлы
    create_sample_files()
    
    # Запускаем графический интерфейс
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()