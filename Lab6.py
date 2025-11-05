import heapq
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional

class Graph:
    """Класс для представления ориентированного взвешенного графа"""
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.adj_matrix = None
        self.num_vertices = 0
        self.is_directed = True
    
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
                for j in range(self.num_vertices):
                    if self.adj_matrix[i][j] != float('inf') and i != j:
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
            # Диагональ заполняем нулями
            for i in range(self.num_vertices):
                self.adj_matrix[i][i] = 0
            
            print(f"Граф загружен: {self.num_vertices} вершин, {len(self.edges)} ребер")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False
    
    def get_neighbors(self, vertex: int) -> List[Tuple[int, float]]:
        """Получить соседей вершины с весами рёбер"""
        neighbors = []
        for v in range(self.num_vertices):
            if self.adj_matrix[vertex][v] != float('inf') and vertex != v:
                neighbors.append((v, self.adj_matrix[vertex][v]))
        return neighbors

class DijkstraAlgorithm:
    """Реализация алгоритма Дейкстры"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.distances = []
        self.predecessors = []
        self.execution_time = 0
    
    def find_shortest_path(self, start: int, end: int) -> Tuple[float, List[int]]:
        """Поиск кратчайшего пути от start до end"""
        start_time = time.time()
        
        # Инициализация
        self.distances = [float('inf')] * self.graph.num_vertices
        self.predecessors = [-1] * self.graph.num_vertices
        self.distances[start] = 0
        
        # Приоритетная очередь (расстояние, вершина)
        pq = [(0, start)]
        
        while pq:
            current_dist, current_vertex = heapq.heappop(pq)
            
            # Если достигли конечной вершины, можно выйти
            if current_vertex == end:
                break
            
            # Если текущее расстояние больше найденного, пропускаем
            if current_dist > self.distances[current_vertex]:
                continue
            
            # Обход соседей
            for neighbor, weight in self.graph.get_neighbors(current_vertex):
                distance = current_dist + weight
                
                # Если нашли более короткий путь
                if distance < self.distances[neighbor]:
                    self.distances[neighbor] = distance
                    self.predecessors[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        # Восстановление пути
        path = self._reconstruct_path(start, end)
        return self.distances[end], path
    
    def _reconstruct_path(self, start: int, end: int) -> List[int]:
        """Восстановление пути от start до end"""
        if self.distances[end] == float('inf'):
            return []  # Пути не существует
        
        path = []
        current = end
        
        while current != -1:
            path.append(current)
            current = self.predecessors[current]
        
        path.reverse()
        
        # Проверяем, что путь начинается с start
        if path[0] == start:
            return path
        else:
            return []  # Пути не существует
    
    def get_all_distances(self, start: int) -> List[float]:
        """Получить расстояния от start до всех вершин"""
        self.find_shortest_path(start, self.graph.num_vertices - 1)  # Запускаем для любой конечной вершины
        return self.distances
    
    def get_execution_time(self) -> float:
        """Возвращает время выполнения алгоритма"""
        return self.execution_time

class GraphVisualizer:
    """Класс для визуализации графа"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def draw_graph(self, graph: Graph, shortest_path: List[int] = None, 
                   start: int = None, end: int = None, title: str = "Граф"):
        """Отрисовка графа с выделением кратчайшего пути"""
        G = nx.DiGraph() if graph.is_directed else nx.Graph()
        
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
                              width=1, alpha=0.5, arrows=True,
                              arrowstyle='->', arrowsize=20)
        
        # Рисуем кратчайший путь красным цветом
        if shortest_path and len(shortest_path) > 1:
            path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=path_edges, 
                                  edge_color='red', width=3, alpha=0.8,
                                  arrowstyle='->', arrowsize=25)
        
        # Рисуем вершины
        node_colors = []
        for i in range(graph.num_vertices):
            if i == start:
                node_colors.append('green')  # Начальная вершина - зеленая
            elif i == end:
                node_colors.append('blue')   # Конечная вершина - синяя
            elif shortest_path and i in shortest_path:
                node_colors.append('orange') # Вершины пути - оранжевые
            else:
                node_colors.append('lightblue') # Остальные - голубые
        
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color=node_colors, 
                              node_size=600, alpha=0.9)
        
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
        ax1.set_title('Зависимость времени от числа вершин\n(Алгоритм Дейкстры)')
        ax1.grid(True, alpha=0.3)
        
        # График зависимости от числа ребер
        edges = list(complexity_data['by_edges'].keys())
        times_edges = list(complexity_data['by_edges'].values())
        
        ax2.plot(edges, times_edges, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Количество ребер')
        ax2.set_ylabel('Время выполнения (секунды)')
        ax2.set_title('Зависимость времени от числа ребер\n(Алгоритм Дейкстры)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class DijkstraApp:
    """Графический интерфейс для работы с алгоритмом Дейкстры"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритм Дейкстры - Поиск кратчайшего пути")
        self.root.geometry("900x700")
        
        self.graph = Graph()
        self.dijkstra = None
        self.visualizer = GraphVisualizer()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Алгоритм Дейкстры - Поиск кратчайшего пути", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Фрейм загрузки графа
        load_frame = ttk.LabelFrame(main_frame, text="Загрузка графа", padding="10")
        load_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(load_frame, text="Загрузить матрицу смежности", 
                  command=self.load_adjacency_matrix).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(load_frame, text="Загрузить список ребер", 
                  command=self.load_edges_list).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(load_frame, text="Сгенерировать тестовый граф", 
                  command=self.generate_test_graph).grid(row=0, column=2)
        
        # Информация о графе
        self.info_label = ttk.Label(main_frame, text="Граф не загружен", 
                                   font=('Arial', 10))
        self.info_label.grid(row=2, column=0, columnspan=4, pady=(0, 10))
        
        # Фрейм выбора вершин
        vertex_frame = ttk.LabelFrame(main_frame, text="Выбор вершин", padding="10")
        vertex_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(vertex_frame, text="Начальная вершина:").grid(row=0, column=0, padx=(0, 10))
        self.start_var = tk.StringVar()
        self.start_combo = ttk.Combobox(vertex_frame, textvariable=self.start_var, width=10)
        self.start_combo.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(vertex_frame, text="Конечная вершина:").grid(row=0, column=2, padx=(0, 10))
        self.end_var = tk.StringVar()
        self.end_combo = ttk.Combobox(vertex_frame, textvariable=self.end_var, width=10)
        self.end_combo.grid(row=0, column=3, padx=(0, 20))
        
        # Фрейм алгоритма
        algo_frame = ttk.LabelFrame(main_frame, text="Алгоритм Дейкстры", padding="10")
        algo_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(algo_frame, text="Найти кратчайший путь", 
                  command=self.find_shortest_path).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Показать все расстояния", 
                  command=self.show_all_distances).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Визуализировать граф", 
                  command=self.visualize_graph).grid(row=0, column=2, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Анализ сложности", 
                  command=self.analyze_complexity).grid(row=0, column=3)
        
        # Фрейм результатов
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        result_frame.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Текстовое поле для вывода результатов
        self.result_text = tk.Text(result_frame, height=20, width=100)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Настройка весов строк и столбцов для растягивания
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def update_vertex_comboboxes(self):
        """Обновление выпадающих списков вершин"""
        if self.graph.num_vertices > 0:
            vertices = [str(i) for i in range(self.graph.num_vertices)]
            self.start_combo['values'] = vertices
            self.end_combo['values'] = vertices
            if vertices:
                self.start_var.set(vertices[0])
                self.end_var.set(vertices[-1])
    
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
                self.dijkstra = DijkstraAlgorithm(self.graph)
                self.update_vertex_comboboxes()
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
                self.dijkstra = DijkstraAlgorithm(self.graph)
                self.update_vertex_comboboxes()
                self.log_message(f"Граф загружен из файла: {filename}")
    
    def generate_test_graph(self):
        """Генерация тестового графа"""
        # Создаем ориентированный граф с 6 вершинами
        self.graph.num_vertices = 6
        self.graph.vertices = list(range(6))
        self.graph.edges = []
        
        # Добавляем случайные ориентированные ребра с весами
        import random
        edges_to_add = 12  # Примерное количество ребер
        
        while len(self.graph.edges) < edges_to_add:
            u = random.randint(0, 5)
            v = random.randint(0, 5)
            if u != v:  # Петли не допускаем
                weight = random.uniform(1, 10)
                self.graph.edges.append((u, v, weight))
        
        # Создаем матрицу смежности
        self.graph.adj_matrix = np.full((6, 6), float('inf'))
        for u, v, weight in self.graph.edges:
            self.graph.adj_matrix[u][v] = weight
        for i in range(6):
            self.graph.adj_matrix[i][i] = 0
        
        self.info_label.config(
            text=f"Тестовый граф: {self.graph.num_vertices} вершин, {len(self.graph.edges)} ребер"
        )
        self.dijkstra = DijkstraAlgorithm(self.graph)
        self.update_vertex_comboboxes()
        self.log_message("Сгенерирован тестовый ориентированный граф (6 вершин)")
    
    def find_shortest_path(self):
        """Поиск кратчайшего пути"""
        if not self.dijkstra:
            messagebox.showerror("Ошибка", "Сначала загрузите граф")
            return
        
        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())
            
            if start == end:
                messagebox.showwarning("Предупреждение", "Начальная и конечная вершины совпадают")
                return
            
            distance, path = self.dijkstra.find_shortest_path(start, end)
            execution_time = self.dijkstra.get_execution_time()
            
            self.log_message("\n" + "="*60)
            self.log_message("ПОИСК КРАТЧАЙШЕГО ПУТИ (Алгоритм Дейкстры)")
            self.log_message("="*60)
            self.log_message(f"Начальная вершина: {start}")
            self.log_message(f"Конечная вершина: {end}")
            
            if path:
                self.log_message(f"Кратчайшее расстояние: {distance:.2f}")
                self.log_message(f"Путь: {' -> '.join(map(str, path))}")
                self.log_message(f"Время выполнения: {execution_time:.6f} секунд")
                
                # Визуализация результата
                self.visualizer.draw_graph(self.graph, path, start, end,
                                         f"Кратчайший путь: {start} → {end} (расстояние: {distance:.2f})")
            else:
                self.log_message("Путь не существует!")
                self.visualizer.draw_graph(self.graph, None, start, end,
                                         f"Путь не существует: {start} → {end}")
            
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный номер вершины")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении алгоритма: {e}")
    
    def show_all_distances(self):
        """Показать расстояния от начальной вершины до всех остальных"""
        if not self.dijkstra:
            messagebox.showerror("Ошибка", "Сначала загрузите граф")
            return
        
        try:
            start = int(self.start_var.get())
            distances = self.dijkstra.get_all_distances(start)
            
            self.log_message("\n" + "="*50)
            self.log_message(f"РАССТОЯНИЯ ОТ ВЕРШИНЫ {start} ДО ВСЕХ ОСТАЛЬНЫХ")
            self.log_message("="*50)
            
            for i, dist in enumerate(distances):
                if i != start:
                    if dist == float('inf'):
                        self.log_message(f"{start} → {i}: недостижима")
                    else:
                        self.log_message(f"{start} → {i}: {dist:.2f}")
            
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный номер вершины")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении алгоритма: {e}")
    
    def visualize_graph(self):
        """Визуализация графа"""
        if not self.graph or self.graph.num_vertices == 0:
            messagebox.showerror("Ошибка", "Сначала загрузите граф")
            return
        
        try:
            start = int(self.start_var.get()) if self.start_var.get() else None
            end = int(self.end_var.get()) if self.end_var.get() else None
            self.visualizer.draw_graph(self.graph, None, start, end, "Исходный граф")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при визуализации: {e}")
    
    def analyze_complexity(self):
        """Анализ временной сложности алгоритма"""
        complexity_data = self._measure_complexity()
        
        # Вывод результатов в текстовое поле
        self.log_message("\n" + "="*60)
        self.log_message("АНАЛИЗ ВРЕМЕННОЙ СЛОЖНОСТИ (Алгоритм Дейкстры)")
        self.log_message("="*60)
        
        self.log_message("\nЗависимость от числа вершин (при V*(V-1) рёбрах):")
        self.log_message("Вершины | Время (сек)")
        self.log_message("-" * 25)
        for vertices, time_val in complexity_data['by_vertices'].items():
            self.log_message(f"{vertices:7} | {time_val:.6f}")
        
        self.log_message("\nЗависимость от числа ребер (при 10 вершинах):")
        self.log_message("Ребра   | Время (сек)")
        self.log_message("-" * 25)
        for edges, time_val in complexity_data['by_edges'].items():
            self.log_message(f"{edges:7} | {time_val:.6f}")
        
        # Визуализация графиков
        self.visualizer.draw_complexity_chart(complexity_data)
    
    def _measure_complexity(self) -> Dict:
        """Измерение временной сложности на различных графах"""
        complexity_data = {'by_vertices': {}, 'by_edges': {}}
        
        # Тестирование зависимости от числа вершин
        print("Измерение зависимости от числа вершин...")
        for n in [10, 20, 30, 40, 50]:
            test_graph = self._generate_test_graph_by_vertices(n)
            dijkstra = DijkstraAlgorithm(test_graph)
            # Замеряем время поиска пути от 0 до n-1
            distance, path = dijkstra.find_shortest_path(0, n-1)
            complexity_data['by_vertices'][n] = dijkstra.get_execution_time()
        
        # Тестирование зависимости от числа ребер
        print("Измерение зависимости от числа ребер...")
        for e in [20, 40, 60, 80, 100]:
            test_graph = self._generate_test_graph_by_edges(10, e)  # 10 вершин
            dijkstra = DijkstraAlgorithm(test_graph)
            distance, path = dijkstra.find_shortest_path(0, 9)
            complexity_data['by_edges'][e] = dijkstra.get_execution_time()
        
        return complexity_data
    
    def _generate_test_graph_by_vertices(self, n: int) -> Graph:
        """Генерация тестового графа с n вершинами"""
        graph = Graph()
        graph.num_vertices = n
        graph.vertices = list(range(n))
        graph.edges = []
        
        import random
        # Создаем связный ориентированный граф
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < 0.3:  # 30% вероятность добавления ребра
                    weight = random.uniform(1, 10)
                    graph.edges.append((i, j, weight))
        
        # Создаем матрицу смежности
        graph.adj_matrix = np.full((n, n), float('inf'))
        for u, v, weight in graph.edges:
            graph.adj_matrix[u][v] = weight
        for i in range(n):
            graph.adj_matrix[i][i] = 0
        
        return graph
    
    def _generate_test_graph_by_edges(self, n: int, e: int) -> Graph:
        """Генерация тестового графа с n вершинами и e ребрами"""
        graph = Graph()
        graph.num_vertices = n
        graph.vertices = list(range(n))
        graph.edges = []
        
        import random
        # Добавляем рёбра случайным образом
        while len(graph.edges) < e:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not any((u, v, w) in graph.edges for w in [0]):
                graph.edges.append((u, v, random.uniform(1, 10)))
        
        # Создаем матрицу смежности
        graph.adj_matrix = np.full((n, n), float('inf'))
        for u, v, weight in graph.edges:
            graph.adj_matrix[u][v] = weight
        for i in range(n):
            graph.adj_matrix[i][i] = 0
        
        return graph
    
    def log_message(self, message: str):
        """Добавление сообщения в текстовое поле"""
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.root.update()

def create_sample_files():
    """Создание примерных файлов для тестирования"""
    
    # Пример матрицы смежности ориентированного графа (5 вершин)
    adjacency_matrix = """0 2 inf 6 inf
inf 0 3 inf inf
inf inf 0 inf 7
inf 8 inf 0 9
inf inf inf inf 0"""
    
    with open('sample_dijkstra_adjacency.txt', 'w') as f:
        f.write(adjacency_matrix)
    
    # Пример списка ребер ориентированного графа
    edges_list = """0 1 2
0 3 6
1 2 3
2 4 7
3 1 8
3 4 9"""
    
    with open('sample_dijkstra_edges.txt', 'w') as f:
        f.write(edges_list)
    
    print("Созданы примерные файлы: sample_dijkstra_adjacency.txt, sample_dijkstra_edges.txt")

def main():
    # Создаем примерные файлы
    create_sample_files()
    
    # Запускаем графический интерфейс
    root = tk.Tk()
    app = DijkstraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()