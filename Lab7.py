import heapq
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Set
import math

class Grid:
    """Класс для представления сетки с стоимостями"""
    def __init__(self):
        self.grid = None
        self.rows = 0
        self.cols = 0
        self.start = (0, 0)
        self.goal = (0, 0)
    
    def load_from_file(self, filename: str):
        """Загрузка сетки из файла"""
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            grid_data = []
            for line in lines:
                if line.strip():
                    row = [float(x) for x in line.strip().split()]
                    grid_data.append(row)
            
            self.grid = np.array(grid_data)
            self.rows, self.cols = self.grid.shape
            self.start = (0, 0)
            self.goal = (self.rows - 1, self.cols - 1)
            
            print(f"Сетка загружена: {self.rows}x{self.cols}")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False
    
    def generate_random_grid(self, rows: int, cols: int):
        """Генерация случайной сетки"""
        self.rows = rows
        self.cols = cols
        self.grid = np.random.uniform(1, 10, (rows, cols))
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        print(f"Сгенерирована случайная сетка: {rows}x{cols}")
    
    def is_valid_cell(self, row: int, col: int) -> bool:
        """Проверка валидности клетки"""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_cost(self, row: int, col: int) -> float:
        """Получение стоимости клетки"""
        if self.is_valid_cell(row, col):
            return self.grid[row][col]
        return float('inf')
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Получение ортогональных соседей"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Вверх, вниз, влево, вправо
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_cell(new_row, new_col):
                neighbors.append((new_row, new_col))
        
        return neighbors

class OrthogonalBeamAlgorithm:
    """Ортогональный лучевой алгоритм (жадный)"""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.visited = set()
        self.path = []
        self.considered_cells = set()
        self.total_cost = 0
        self.execution_time = 0
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """Поиск пути ортогональным лучевым алгоритмом"""
        start_time = time.time()
        
        self.visited = set()
        self.path = []
        self.considered_cells = set()
        self.total_cost = 0
        
        current = start
        self.path.append(current)
        self.visited.add(current)
        self.considered_cells.add(current)
        
        while current != goal:
            # Получаем всех ортогональных соседей
            neighbors = self.grid.get_neighbors(current[0], current[1])
            
            # Фильтруем непосещенных соседей
            unvisited_neighbors = [n for n in neighbors if n not in self.visited]
            
            if not unvisited_neighbors:
                # Если нет непосещенных соседей, откатываемся
                if len(self.path) > 1:
                    self.path.pop()
                    current = self.path[-1]
                else:
                    break  # Путь не найден
                continue
            
            # Жадный выбор: выбираем соседа с минимальной стоимостью
            best_neighbor = min(unvisited_neighbors, 
                              key=lambda n: self.grid.get_cost(n[0], n[1]))
            
            # Добавляем в рассмотренные клетки
            for neighbor in unvisited_neighbors:
                self.considered_cells.add(neighbor)
            
            # Переходим к лучшему соседу
            current = best_neighbor
            self.path.append(current)
            self.visited.add(current)
            self.total_cost += self.grid.get_cost(current[0], current[1])
            
            # Ограничение на длину пути для избежания бесконечных циклов
            if len(self.path) > self.grid.rows * self.grid.cols * 2:
                break
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        # Проверяем, достигли ли цели
        if self.path[-1] != goal:
            return [], float('inf')
        
        return self.path, self.total_cost
    
    def get_execution_time(self) -> float:
        return self.execution_time
    
    def get_considered_cells(self) -> Set[Tuple[int, int]]:
        return self.considered_cells

class AStarAlgorithm:
    """Алгоритм A*"""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.visited = set()
        self.came_from = {}
        self.g_score = {}
        self.f_score = {}
        self.considered_cells = set()
        self.execution_time = 0
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Эвристическая функция (манхэттенское расстояние)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """Поиск пути алгоритмом A*"""
        start_time = time.time()
        
        self.visited = set()
        self.came_from = {}
        self.considered_cells = set()
        
        # Инициализация оценок
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        
        open_set = [(self.f_score[start], start)]
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            self.considered_cells.add(current)
            
            if current == goal:
                # Восстанавливаем путь
                path = self._reconstruct_path(current)
                total_cost = self.g_score[goal]
                end_time = time.time()
                self.execution_time = end_time - start_time
                return path, total_cost
            
            self.visited.add(current)
            
            for neighbor in self.grid.get_neighbors(current[0], current[1]):
                if neighbor in self.visited:
                    continue
                
                tentative_g_score = self.g_score[current] + self.grid.get_cost(neighbor[0], neighbor[1])
                
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (self.f_score[neighbor], neighbor))
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        return [], float('inf')
    
    def _reconstruct_path(self, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Восстановление пути"""
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def get_execution_time(self) -> float:
        return self.execution_time
    
    def get_considered_cells(self) -> Set[Tuple[int, int]]:
        return self.considered_cells

class GridVisualizer:
    """Класс для визуализации сетки и путей"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def draw_grid(self, grid: Grid, path: List[Tuple[int, int]] = None, 
                 considered_cells: Set[Tuple[int, int]] = None,
                 start: Tuple[int, int] = None, goal: Tuple[int, int] = None,
                 title: str = "Сетка", algorithm_name: str = ""):
        """Отрисовка сетки с путём и рассмотренными клетками"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Отображаем сетку с цветовой кодировкой стоимости
        im = self.ax.imshow(grid.grid, cmap='YlOrRd', alpha=0.7)
        plt.colorbar(im, ax=self.ax, label='Стоимость клетки')
        
        # Выделяем рассмотренные клетки
        if considered_cells:
            for row, col in considered_cells:
                self.ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, 
                                              fill=True, color='lightblue', alpha=0.5))
        
        # Рисуем путь
        if path and len(path) > 1:
            path_cols = [cell[1] for cell in path]
            path_rows = [cell[0] for cell in path]
            self.ax.plot(path_cols, path_rows, 'b-', linewidth=3, alpha=0.8, label='Путь')
            self.ax.scatter(path_cols, path_rows, c='blue', s=50, alpha=0.8)
        
        # Отмечаем старт и финиш
        if start:
            self.ax.plot(start[1], start[0], 'gs', markersize=15, label='Старт', markeredgecolor='black')
        if goal:
            self.ax.plot(goal[1], goal[0], 'rs', markersize=15, label='Финиш', markeredgecolor='black')
        
        # Добавляем значения стоимости в клетки
        for i in range(grid.rows):
            for j in range(grid.cols):
                self.ax.text(j, i, f'{grid.grid[i][j]:.1f}', 
                           ha='center', va='center', fontsize=8, 
                           fontweight='bold', color='black')
        
        self.ax.set_xticks(range(grid.cols))
        self.ax.set_yticks(range(grid.rows))
        self.ax.grid(True, color='black', alpha=0.3)
        self.ax.set_title(f'{title}\n{algorithm_name}', fontsize=14, fontweight='bold')
        self.ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def draw_comparison(self, grid: Grid, results: Dict):
        """Отрисовка сравнения алгоритмов"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Исходная сетка
        im0 = axes[0].imshow(grid.grid, cmap='YlOrRd')
        axes[0].set_title('Исходная сетка')
        axes[0].grid(True, color='black', alpha=0.3)
        plt.colorbar(im0, ax=axes[0])
        
        # Ортогональный лучевой
        beam_result = results['beam']
        if beam_result['path']:
            path_cols = [cell[1] for cell in beam_result['path']]
            path_rows = [cell[0] for cell in beam_result['path']]
            axes[1].plot(path_cols, path_rows, 'b-', linewidth=3)
            axes[1].scatter(path_cols, path_rows, c='blue', s=30)
        
        # Выделяем рассмотренные клетки
        for row, col in beam_result['considered_cells']:
            axes[1].add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, 
                                          fill=True, color='lightblue', alpha=0.5))
        
        axes[1].imshow(grid.grid, cmap='YlOrRd', alpha=0.3)
        axes[1].set_title(f'Ортогональный лучевой\nСтоимость: {beam_result["cost"]:.2f}\nВремя: {beam_result["time"]:.6f}с')
        axes[1].grid(True, color='black', alpha=0.3)
        
        # A*
        astar_result = results['astar']
        if astar_result['path']:
            path_cols = [cell[1] for cell in astar_result['path']]
            path_rows = [cell[0] for cell in astar_result['path']]
            axes[2].plot(path_cols, path_rows, 'r-', linewidth=3)
            axes[2].scatter(path_cols, path_rows, c='red', s=30)
        
        # Выделяем рассмотренные клетки
        for row, col in astar_result['considered_cells']:
            axes[2].add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, 
                                          fill=True, color='lightgreen', alpha=0.5))
        
        axes[2].imshow(grid.grid, cmap='YlOrRd', alpha=0.3)
        axes[2].set_title(f'A*\nСтоимость: {astar_result["cost"]:.2f}\nВремя: {astar_result["time"]:.6f}с')
        axes[2].grid(True, color='black', alpha=0.3)
        
        # Отмечаем старт и финиш на всех графиках
        for ax in axes:
            ax.plot(grid.start[1], grid.start[0], 'gs', markersize=10, markeredgecolor='black')
            ax.plot(grid.goal[1], grid.goal[0], 'rs', markersize=10, markeredgecolor='black')
        
        plt.tight_layout()
        plt.show()
    
    def draw_complexity_chart(self, complexity_data: Dict):
        """Отрисовка графика временной сложности"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График для ортогонального лучевого
        sizes = list(complexity_data['beam'].keys())
        beam_times = [complexity_data['beam'][size] for size in sizes]
        astar_times = [complexity_data['astar'][size] for size in sizes]
        
        ax1.plot(sizes, beam_times, 'bo-', linewidth=2, markersize=6, label='Ортогональный лучевой')
        ax1.plot(sizes, astar_times, 'ro-', linewidth=2, markersize=6, label='A*')
        ax1.set_xlabel('Размер сетки (N x N)')
        ax1.set_ylabel('Время выполнения (секунды)')
        ax1.set_title('Сравнение временной сложности')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График отношения времен
        time_ratio = [beam_times[i] / astar_times[i] if astar_times[i] > 0 else 0 
                     for i in range(len(sizes))]
        
        ax2.plot(sizes, time_ratio, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Размер сетки (N x N)')
        ax2.set_ylabel('Отношение времени (Лучевой / A*)')
        ax2.set_title('Относительная эффективность')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class PathfindingApp:
    """Графический интерфейс для поиска путей на сетке"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Поиск путей на сетке - Ортогональный лучевой vs A*")
        self.root.geometry("1000x800")
        
        self.grid = Grid()
        self.beam_algorithm = None
        self.astar_algorithm = None
        self.visualizer = GridVisualizer()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Заголовок
        title_label = ttk.Label(main_frame, 
                               text="Поиск путей на сетке - Ортогональный лучевой алгоритм vs A*", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Фрейм загрузки сетки
        grid_frame = ttk.LabelFrame(main_frame, text="Загрузка сетки", padding="10")
        grid_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(grid_frame, text="Загрузить сетку из файла", 
                  command=self.load_grid_from_file).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(grid_frame, text="Сгенерировать случайную сетку", 
                  command=self.generate_random_grid).grid(row=0, column=1, padx=(0, 10))
        
        # Фрейм параметров
        param_frame = ttk.LabelFrame(main_frame, text="Параметры", padding="10")
        param_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(param_frame, text="Старт (row,col):").grid(row=0, column=0, padx=(0, 5))
        self.start_var = tk.StringVar(value="0,0")
        ttk.Entry(param_frame, textvariable=self.start_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(param_frame, text="Финиш (row,col):").grid(row=0, column=2, padx=(0, 5))
        self.goal_var = tk.StringVar(value="4,4")
        ttk.Entry(param_frame, textvariable=self.goal_var, width=10).grid(row=0, column=3, padx=(0, 15))
        
        # Информация о сетке
        self.info_label = ttk.Label(main_frame, text="Сетка не загружена", 
                                   font=('Arial', 10))
        self.info_label.grid(row=3, column=0, columnspan=4, pady=(0, 10))
        
        # Фрейм алгоритмов
        algo_frame = ttk.LabelFrame(main_frame, text="Алгоритмы поиска", padding="10")
        algo_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(algo_frame, text="Запустить ортогональный лучевой", 
                  command=self.run_beam_algorithm).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Запустить A*", 
                  command=self.run_astar_algorithm).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(algo_frame, text="Сравнить оба алгоритма", 
                  command=self.compare_algorithms).grid(row=0, column=2, padx=(0, 10))
        
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
        
        # Настройка весов для растягивания
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def load_grid_from_file(self):
        """Загрузка сетки из файла"""
        filename = filedialog.askopenfilename(
            title="Выберите файл с сеткой",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            if self.grid.load_from_file(filename):
                self.info_label.config(
                    text=f"Сетка загружена: {self.grid.rows}x{self.grid.cols}"
                )
                self.beam_algorithm = OrthogonalBeamAlgorithm(self.grid)
                self.astar_algorithm = AStarAlgorithm(self.grid)
                self.log_message(f"Сетка загружена из файла: {filename}")
    
    def generate_random_grid(self):
        """Генерация случайной сетки"""
        try:
            rows = int(tk.simpledialog.askstring("Размер сетки", "Введите количество строк:"))
            cols = int(tk.simpledialog.askstring("Размер сетки", "Введите количество столбцов:"))
            
            if rows > 0 and cols > 0:
                self.grid.generate_random_grid(rows, cols)
                self.info_label.config(
                    text=f"Сгенерирована сетка: {rows}x{cols}"
                )
                self.beam_algorithm = OrthogonalBeamAlgorithm(self.grid)
                self.astar_algorithm = AStarAlgorithm(self.grid)
                self.log_message(f"Сгенерирована случайная сетка {rows}x{cols}")
                
        except (ValueError, TypeError):
            messagebox.showerror("Ошибка", "Некорректный размер сетки")
    
    def parse_coordinates(self, coord_str: str) -> Tuple[int, int]:
        """Парсинг координат из строки"""
        try:
            row, col = map(int, coord_str.split(','))
            return row, col
        except:
            raise ValueError("Некорректный формат координат")
    
    def run_beam_algorithm(self):
        """Запуск ортогонального лучевого алгоритма"""
        if not self.beam_algorithm:
            messagebox.showerror("Ошибка", "Сначала загрузите сетку")
            return
        
        try:
            start = self.parse_coordinates(self.start_var.get())
            goal = self.parse_coordinates(self.goal_var.get())
            
            if not self.grid.is_valid_cell(*start) or not self.grid.is_valid_cell(*goal):
                messagebox.showerror("Ошибка", "Некорректные координаты")
                return
            
            path, cost = self.beam_algorithm.find_path(start, goal)
            execution_time = self.beam_algorithm.get_execution_time()
            considered_cells = self.beam_algorithm.get_considered_cells()
            
            self.log_message("\n" + "="*60)
            self.log_message("ОРТОГОНАЛЬНЫЙ ЛУЧЕВОЙ АЛГОРИМ (жадный)")
            self.log_message("="*60)
            self.log_message(f"Старт: {start}, Финиш: {goal}")
            
            if path:
                self.log_message(f"Найден путь длиной {len(path)} клеток")
                self.log_message(f"Общая стоимость: {cost:.2f}")
                self.log_message(f"Время выполнения: {execution_time:.6f} сек")
                self.log_message(f"Рассмотрено клеток: {len(considered_cells)}")
                self.log_message(f"Путь: {path}")
            else:
                self.log_message("Путь не найден!")
            
            # Визуализация
            self.visualizer.draw_grid(
                self.grid, path, considered_cells, start, goal,
                "Ортогональный лучевой алгоритм",
                f"Стоимость: {cost:.2f}, Время: {execution_time:.6f}с"
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении алгоритма: {e}")
    
    def run_astar_algorithm(self):
        """Запуск алгоритма A*"""
        if not self.astar_algorithm:
            messagebox.showerror("Ошибка", "Сначала загрузите сетку")
            return
        
        try:
            start = self.parse_coordinates(self.start_var.get())
            goal = self.parse_coordinates(self.goal_var.get())
            
            if not self.grid.is_valid_cell(*start) or not self.grid.is_valid_cell(*goal):
                messagebox.showerror("Ошибка", "Некорректные координаты")
                return
            
            path, cost = self.astar_algorithm.find_path(start, goal)
            execution_time = self.astar_algorithm.get_execution_time()
            considered_cells = self.astar_algorithm.get_considered_cells()
            
            self.log_message("\n" + "="*60)
            self.log_message("АЛГОРИТМ A*")
            self.log_message("="*60)
            self.log_message(f"Старт: {start}, Финиш: {goal}")
            
            if path:
                self.log_message(f"Найден путь длиной {len(path)} клеток")
                self.log_message(f"Общая стоимость: {cost:.2f}")
                self.log_message(f"Время выполнения: {execution_time:.6f} сек")
                self.log_message(f"Рассмотрено клеток: {len(considered_cells)}")
                self.log_message(f"Путь: {path}")
            else:
                self.log_message("Путь не найден!")
            
            # Визуализация
            self.visualizer.draw_grid(
                self.grid, path, considered_cells, start, goal,
                "Алгоритм A*",
                f"Стоимость: {cost:.2f}, Время: {execution_time:.6f}с"
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении алгоритма: {e}")
    
    def compare_algorithms(self):
        """Сравнение обоих алгоритмов"""
        if not self.beam_algorithm or not self.astar_algorithm:
            messagebox.showerror("Ошибка", "Сначала загрузите сетку")
            return
        
        try:
            start = self.parse_coordinates(self.start_var.get())
            goal = self.parse_coordinates(self.goal_var.get())
            
            if not self.grid.is_valid_cell(*start) or not self.grid.is_valid_cell(*goal):
                messagebox.showerror("Ошибка", "Некорректные координаты")
                return
            
            # Запуск обоих алгоритмов
            beam_path, beam_cost = self.beam_algorithm.find_path(start, goal)
            beam_time = self.beam_algorithm.get_execution_time()
            beam_considered = self.beam_algorithm.get_considered_cells()
            
            astar_path, astar_cost = self.astar_algorithm.find_path(start, goal)
            astar_time = self.astar_algorithm.get_execution_time()
            astar_considered = self.astar_algorithm.get_considered_cells()
            
            # Вывод результатов
            self.log_message("\n" + "="*70)
            self.log_message("СРАВНЕНИЕ АЛГОРИТМОВ")
            self.log_message("="*70)
            self.log_message(f"Старт: {start}, Финиш: {goal}")
            
            self.log_message("\nОртогональный лучевой:")
            if beam_path:
                self.log_message(f"  Путь найден, стоимость: {beam_cost:.2f}")
                self.log_message(f"  Время: {beam_time:.6f} сек")
                self.log_message(f"  Рассмотрено клеток: {len(beam_considered)}")
            else:
                self.log_message("  Путь не найден")
            
            self.log_message("\nA*:")
            if astar_path:
                self.log_message(f"  Путь найден, стоимость: {astar_cost:.2f}")
                self.log_message(f"  Время: {astar_time:.6f} сек")
                self.log_message(f"  Рассмотрено клеток: {len(astar_considered)}")
            else:
                self.log_message("  Путь не найден")
            
            # Сравнительный анализ
            if beam_path and astar_path:
                self.log_message("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
                self.log_message(f"  Разница в стоимости: {abs(beam_cost - astar_cost):.2f}")
                self.log_message(f"  Разница во времени: {abs(beam_time - astar_time):.6f} сек")
                self.log_message(f"  Разница в рассмотренных клетках: {abs(len(beam_considered) - len(astar_considered))}")
                
                if beam_cost < astar_cost:
                    self.log_message("  Ортогональный лучевой нашел более дешевый путь")
                elif beam_cost > astar_cost:
                    self.log_message("  A* нашел более дешевый путь")
                else:
                    self.log_message("  Оба алгоритма нашли пути одинаковой стоимости")
            
            # Визуализация сравнения
            results = {
                'beam': {
                    'path': beam_path,
                    'cost': beam_cost,
                    'time': beam_time,
                    'considered_cells': beam_considered
                },
                'astar': {
                    'path': astar_path,
                    'cost': astar_cost,
                    'time': astar_time,
                    'considered_cells': astar_considered
                }
            }
            
            self.visualizer.draw_comparison(self.grid, results)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сравнении алгоритмов: {e}")
    
    def analyze_complexity(self):
        """Анализ временной сложности"""
        complexity_data = self._measure_complexity()
        
        # Вывод результатов
        self.log_message("\n" + "="*60)
        self.log_message("АНАЛИЗ ВРЕМЕННОЙ СЛОЖНОСТИ")
        self.log_message("="*60)
        
        self.log_message("\nЗависимость от размера сетки (N x N):")
        self.log_message("Размер | Лучевой (сек) | A* (сек)   | Отношение")
        self.log_message("-" * 55)
        
        for size in sorted(complexity_data['beam'].keys()):
            beam_time = complexity_data['beam'][size]
            astar_time = complexity_data['astar'][size]
            ratio = beam_time / astar_time if astar_time > 0 else 0
            self.log_message(f"{size:6} | {beam_time:12.6f} | {astar_time:10.6f} | {ratio:8.2f}")
        
        # Визуализация графиков
        self.visualizer.draw_complexity_chart(complexity_data)
    
    def _measure_complexity(self) -> Dict:
        """Измерение временной сложности на различных сетках"""
        complexity_data = {'beam': {}, 'astar': {}}
        
        # Тестирование на сетках разных размеров
        sizes = [5, 10, 15, 20, 25]
        
        for size in sizes:
            print(f"Тестирование на сетке {size}x{size}...")
            
            # Создаем тестовую сетку
            test_grid = Grid()
            test_grid.generate_random_grid(size, size)
            
            # Тестируем ортогональный лучевой
            beam_algo = OrthogonalBeamAlgorithm(test_grid)
            beam_path, beam_cost = beam_algo.find_path((0, 0), (size-1, size-1))
            complexity_data['beam'][size] = beam_algo.get_execution_time()
            
            # Тестируем A*
            astar_algo = AStarAlgorithm(test_grid)
            astar_path, astar_cost = astar_algo.find_path((0, 0), (size-1, size-1))
            complexity_data['astar'][size] = astar_algo.get_execution_time()
        
        return complexity_data
    
    def log_message(self, message: str):
        """Добавление сообщения в текстовое поле"""
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.root.update()

def create_sample_grid_file():
    """Создание примерного файла с сеткой"""
    grid_data = """1.0 2.0 1.5 3.0 2.5
2.0 5.0 1.0 4.0 3.5
1.5 1.0 2.0 2.5 1.0
3.0 4.0 2.5 1.0 2.0
2.5 3.5 1.0 2.0 1.5"""
    
    with open('sample_grid.txt', 'w') as f:
        f.write(grid_data)
    
    print("Создан примерный файл: sample_grid.txt")

def main():
    # Создаем примерный файл
    create_sample_grid_file()
    
    # Запускаем графический интерфейс
    root = tk.Tk()
    app = PathfindingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()