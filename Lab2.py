import sys

class MatrixChainMultiplication:
    def __init__(self):
        self.m = None  # Таблица минимальных стоимостей
        self.s = None  # Таблица оптимальных разбиений
    
    def matrix_chain_order(self, p):
        """Алгоритм нахождения оптимального порядка перемножения матриц"""
        n = len(p) - 1  # Количество матриц
        self.m = [[0] * (n + 1) for _ in range(n + 1)]
        self.s = [[0] * (n + 1) for _ in range(n + 1)]
        
        # l - длина цепочки
        for l in range(2, n + 1):
            for i in range(1, n - l + 2):
                j = i + l - 1
                self.m[i][j] = sys.maxsize
                
                for k in range(i, j):
                    # Стоимость перемножения цепочек [i,k] и [k+1,j]
                    cost = self.m[i][k] + self.m[k + 1][j] + p[i - 1] * p[k] * p[j]
                    
                    if cost < self.m[i][j]:
                        self.m[i][j] = cost
                        self.s[i][j] = k
        
        return self.m[1][n]
    
    def print_optimal_parenthesization(self, i, j):
        """Рекурсивный вывод оптимальной расстановки скобок"""
        if i == j:
            return f"A{i}"
        else:
            k = self.s[i][j]
            left = self.print_optimal_parenthesization(i, k)
            right = self.print_optimal_parenthesization(k + 1, j)
            return f"({left} × {right})"
    
    def get_optimal_solution(self, p):
        """Получить полное решение"""
        n = len(p) - 1
        min_cost = self.matrix_chain_order(p)
        optimal_order = self.print_optimal_parenthesization(1, n)
        return min_cost, optimal_order
    
    def print_tables(self, p):
        """Вывод таблиц m и s для анализа"""
        n = len(p) - 1
        
        print("\n" + "="*60)
        print("ТАБЛИЦА m[i][j] (минимальные стоимости)")
        print("="*60)
        print("i\\j", end="")
        for j in range(1, n + 1):
            print(f"{j:8}", end="")
        print()
        
        for i in range(1, n + 1):
            print(f"{i:3}", end="")
            for j in range(1, n + 1):
                if i <= j:
                    print(f"{self.m[i][j]:8}", end="")
                else:
                    print(f"{'':8}", end="")
            print()
        
        print("\n" + "="*60)
        print("ТАБЛИЦА s[i][j] (оптимальные точки разбиения)")
        print("="*60)
        print("i\\j", end="")
        for j in range(1, n + 1):
            print(f"{j:4}", end="")
        print()
        
        for i in range(1, n + 1):
            print(f"{i:3}", end="")
            for j in range(1, n + 1):
                if i < j:
                    print(f"{self.s[i][j]:4}", end="")
                else:
                    print(f"{'':4}", end="")
            print()


def validate_dimensions(p):
    """Проверка корректности размерностей матриц"""
    if len(p) < 2:
        raise ValueError("Должно быть хотя бы 2 размерности для одной матрицы")
    
    for i in range(len(p) - 1):
        if p[i] <= 0 or p[i + 1] <= 0:
            raise ValueError("Размерности матриц должны быть положительными числами")
    
    return True


def main():
    mcm = MatrixChainMultiplication()
    
    while True:
        print("\n" + "="*70)
        print("Лабораторная работа №2. Перемножение цепочки матриц")
        print("="*70)
        print("1. Ввести размерности матриц вручную")
        print("2. Использовать пример из учебника")
        print("3. Сгенерировать случайный пример")
        print("4. Выход")
        print("-"*70)
        
        choice = input("Выберите пункт меню (1-4): ").strip()
        
        if choice == '1':
            try:
                print("\nВведите размерности матриц через пробел")
                print("Пример: для матриц 10×20, 20×50, 50×1 введите: 10 20 50 1")
                input_str = input("Размерности: ").strip()
                
                p = list(map(int, input_str.split()))
                
                if validate_dimensions(p):
                    min_cost, optimal_order = mcm.get_optimal_solution(p)
                    
                    print("\n" + "="*50)
                    print("РЕЗУЛЬТАТЫ")
                    print("="*50)
                    print(f"Размерности матриц: {p}")
                    print(f"Минимальное количество скалярных умножений: {min_cost}")
                    print(f"Оптимальный порядок перемножения: {optimal_order}")
                    
                    # Вывод таблиц для анализа
                    show_tables = input("\nПоказать таблицы динамического программирования? (y/n): ").strip().lower()
                    if show_tables == 'y':
                        mcm.print_tables(p)
                        
            except ValueError as e:
                print(f"Ошибка ввода: {e}")
            except Exception as e:
                print(f"Ошибка: {e}")
        
        elif choice == '2':
            # Пример из книги Кормена (стр. 402)
            p = [30, 35, 15, 5, 10, 20, 25]  # 6 матриц
            
            print(f"\nПример из учебника Кормена:")
            print(f"Размерности: {p}")
            print("Матрицы: A1(30×35), A2(35×15), A3(15×5), A4(5×10), A5(10×20), A6(20×25)")
            
            min_cost, optimal_order = mcm.get_optimal_solution(p)
            
            print("\n" + "="*50)
            print("РЕЗУЛЬТАТЫ")
            print("="*50)
            print(f"Минимальное количество скалярных умножений: {min_cost}")
            print(f"Оптимальный порядок перемножения: {optimal_order}")
            
            mcm.print_tables(p)
        
        elif choice == '3':
            import random
            
            n = random.randint(3, 8)  # Количество матриц от 3 до 8
            p = [random.randint(5, 100) for _ in range(n + 1)]
            
            print(f"\nСгенерирован пример с {n} матрицами:")
            print(f"Размерности: {p}")
            for i in range(n):
                print(f"A{i+1}({p[i]}×{p[i+1]})", end=" ")
            print()
            
            min_cost, optimal_order = mcm.get_optimal_solution(p)
            
            print("\n" + "="*50)
            print("РЕЗУЛЬТАТЫ")
            print("="*50)
            print(f"Минимальное количество скалярных умножений: {min_cost}")
            print(f"Оптимальный порядок перемножения: {optimal_order}")
            
            show_tables = input("\nПоказать таблицы динамического программирования? (y/n): ").strip().lower()
            if show_tables == 'y':
                mcm.print_tables(p)
        
        elif choice == '4':
            print("Выход из программы.")
            break
        
        else:
            print("Неверный выбор. Пожалуйста, выберите пункт от 1 до 4.")


def run_tests():
    """Функция для автоматического тестирования"""
    mcm = MatrixChainMultiplication()
    
    test_cases = [
        {
            "name": "Пример из Кормена",
            "p": [30, 35, 15, 5, 10, 20, 25],
            "expected_cost": 15125
        },
        {
            "name": "3 матрицы",
            "p": [10, 20, 30],
            "expected_cost": 6000
        },
        {
            "name": "4 матрицы",
            "p": [5, 10, 15, 20],
            "expected_cost": 1750
        }
    ]
    
    print("\n" + "="*70)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ")
    print("="*70)
    
    for test in test_cases:
        try:
            min_cost, optimal_order = mcm.get_optimal_solution(test["p"])
            status = "✓" if min_cost == test["expected_cost"] else "✗"
            
            print(f"\n{status} {test['name']}")
            print(f"   Размерности: {test['p']}")
            print(f"   Полученная стоимость: {min_cost}")
            print(f"   Ожидаемая стоимость: {test['expected_cost']}")
            print(f"   Оптимальный порядок: {optimal_order}")
            
        except Exception as e:
            print(f"✗ {test['name']}: Ошибка - {e}")


if __name__ == "__main__":
    print("="*70)
    print("АЛГОРИТМ ПЕРЕМНОЖЕНИЯ ЦЕПОЧКИ МАТРИЦ")
    print("Динамическое программирование - вариант 2")
    print("="*70)
    
    # Запуск автоматического тестирования
    run_tests()
    
    # Запуск основного меню
    main()