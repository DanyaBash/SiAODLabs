import time
import json
from typing import List, Tuple

class Item:
    """Класс Предмет"""
    def __init__(self, name: str, weight: float, value: float):
        self.name = name
        self.weight = weight
        self.value = value
    
    @property
    def value_per_weight(self) -> float:
        """Соотношение цена/вес"""
        return self.value / self.weight if self.weight > 0 else 0
    
    def __str__(self):
        return f"{self.name} (вес: {self.weight}, ценность: {self.value}, цена/вес: {self.value_per_weight:.2f})"
    
    def to_dict(self):
        return {
            'name': self.name,
            'weight': self.weight,
            'value': self.value
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['name'], data['weight'], data['value'])

class Knapsack:
    """Класс Рюкзак"""
    def __init__(self, max_weight: float):
        self.max_weight = max_weight
        self.current_weight = 0.0
        self.items = []
        self.total_value = 0.0
    
    def add_item(self, item: Item) -> bool:
        """Добавить предмет в рюкзак"""
        if self.current_weight + item.weight <= self.max_weight:
            self.items.append(item)
            self.current_weight += item.weight
            self.total_value += item.value
            return True
        return False
    
    def clear(self):
        """Очистить рюкзак"""
        self.items.clear()
        self.current_weight = 0.0
        self.total_value = 0.0
    
    def __str__(self):
        if not self.items:
            return "Рюкзак пуст"
        
        result = f"Рюкзак (вес: {self.current_weight}/{self.max_weight}, ценность: {self.total_value}):\n"
        for i, item in enumerate(self.items, 1):
            result += f"  {i}. {item}\n"
        return result

class KnapsackSolver:
    """Класс для решения задачи о рюкзаке разными методами"""
    
    def __init__(self, items: List[Item], max_weight: float):
        self.items = items
        self.max_weight = max_weight
        self.recursion_calls = 0
    
    def recursive_solution(self) -> Knapsack:
        """Рекурсивное решение"""
        self.recursion_calls = 0
        n = len(self.items)
        
        def recursive_knapsack(i: int, remaining_weight: float) -> Tuple[float, List[Item]]:
            self.recursion_calls += 1
            
            if i == 0 or remaining_weight == 0:
                return 0, []
            
            current_item = self.items[i-1]
            
            # Если текущий предмет не помещается
            if current_item.weight > remaining_weight:
                return recursive_knapsack(i-1, remaining_weight)
            
            # Выбираем максимум из двух вариантов: брать предмет или не брать
            value_without, items_without = recursive_knapsack(i-1, remaining_weight)
            value_with, items_with = recursive_knapsack(i-1, remaining_weight - current_item.weight)
            value_with += current_item.value
            
            if value_with > value_without:
                items_with.append(current_item)
                return value_with, items_with
            else:
                return value_without, items_without
        
        total_value, selected_items = recursive_knapsack(n, self.max_weight)
        
        knapsack = Knapsack(self.max_weight)
        for item in selected_items:
            knapsack.add_item(item)
        
        return knapsack, self.recursion_calls
    
    def dynamic_programming_solution(self) -> Knapsack:
        """Решение методом динамического программирования"""
        n = len(self.items)
        # Создаем таблицу DP
        dp = [[0] * (self.max_weight + 1) for _ in range(n + 1)]
        
        # Заполняем таблицу DP
        for i in range(1, n + 1):
            for w in range(1, self.max_weight + 1):
                current_item = self.items[i-1]
                
                if current_item.weight <= w:
                    dp[i][w] = max(dp[i-1][w], 
                                  dp[i-1][int(w - current_item.weight)] + current_item.value)
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Восстанавливаем решение
        knapsack = Knapsack(self.max_weight)
        w = self.max_weight
        
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                item = self.items[i-1]
                knapsack.add_item(item)
                w -= int(item.weight)
        
        return knapsack
    
    def greedy_solution_by_weight(self) -> Knapsack:
        """Жадный алгоритм: предметы с максимальным весом первыми"""
        # Сортируем по убыванию веса
        sorted_items = sorted(self.items, key=lambda x: x.weight, reverse=True)
        
        knapsack = Knapsack(self.max_weight)
        for item in sorted_items:
            knapsack.add_item(item)
        
        return knapsack
    
    def greedy_solution_by_value_per_weight(self) -> Knapsack:
        """Жадный алгоритм: предметы с максимальным соотношением цена/вес первыми"""
        # Сортируем по убыванию соотношения цена/вес
        sorted_items = sorted(self.items, key=lambda x: x.value_per_weight, reverse=True)
        
        knapsack = Knapsack(self.max_weight)
        for item in sorted_items:
            knapsack.add_item(item)
        
        return knapsack

class KnapsackManager:
    """Менеджер для работы с рюкзаком и предметами"""
    
    def __init__(self):
        self.items = []
        self.max_weight = 0
        self.knapsack = None
    
    def load_items_from_file(self, filename: str):
        """Загрузка предметов из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            self.items.clear()
            for item_data in data.get('items', []):
                self.items.append(Item.from_dict(item_data))
            
            self.max_weight = data.get('max_weight', 0)
            self.knapsack = Knapsack(self.max_weight)
            
            print(f"Загружено {len(self.items)} предметов, максимальный вес: {self.max_weight}")
            
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
    
    def save_items_to_file(self, filename: str):
        """Сохранение предметов в файл"""
        try:
            data = {
                'max_weight': self.max_weight,
                'items': [item.to_dict() for item in self.items]
            }
            
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            
            print(f"Данные сохранены в файл: {filename}")
            
        except Exception as e:
            print(f"Ошибка сохранения файла: {e}")
    
    def add_item(self):
        """Добавление предмета"""
        try:
            name = input("Введите название предмета: ").strip()
            weight = float(input("Введите вес предмета: "))
            value = float(input("Введите ценность предмета: "))
            
            if weight <= 0 or value < 0:
                print("Вес должен быть положительным, ценность - неотрицательной")
                return
            
            self.items.append(Item(name, weight, value))
            print(f"Предмет '{name}' добавлен")
            
        except ValueError:
            print("Ошибка ввода данных")
    
    def edit_item(self):
        """Изменение предмета"""
        self.show_items()
        if not self.items:
            return
        
        try:
            index = int(input("Введите номер предмета для изменения: ")) - 1
            if 0 <= index < len(self.items):
                item = self.items[index]
                print(f"Текущие данные: {item}")
                
                name = input(f"Введите новое название [{item.name}]: ").strip()
                weight = input(f"Введите новый вес [{item.weight}]: ").strip()
                value = input(f"Введите новую ценность [{item.value}]: ").strip()
                
                if name:
                    item.name = name
                if weight:
                    item.weight = float(weight)
                if value:
                    item.value = float(value)
                
                print("Предмет изменен")
            else:
                print("Неверный номер предмета")
                
        except (ValueError, IndexError):
            print("Ошибка ввода данных")
    
    def delete_item(self):
        """Удаление предмета"""
        self.show_items()
        if not self.items:
            return
        
        try:
            index = int(input("Введите номер предмета для удаления: ")) - 1
            if 0 <= index < len(self.items):
                item = self.items.pop(index)
                print(f"Предмет '{item.name}' удален")
            else:
                print("Неверный номер предмета")
                
        except (ValueError, IndexError):
            print("Ошибка ввода данных")
    
    def set_max_weight(self):
        """Установка максимального веса рюкзака"""
        try:
            weight = float(input("Введите максимальный вес рюкзака: "))
            if weight <= 0:
                print("Вес должен быть положительным")
                return
            
            self.max_weight = weight
            self.knapsack = Knapsack(self.max_weight)
            print(f"Максимальный вес установлен: {weight}")
            
        except ValueError:
            print("Ошибка ввода данных")
    
    def show_knapsack(self):
        """Просмотр содержимого рюкзака"""
        if self.knapsack:
            print(self.knapsack)
        else:
            print("Рюкзак не инициализирован")
    
    def show_items(self):
        """Просмотр списка предметов"""
        if not self.items:
            print("Список предметов пуст")
            return
        
        print(f"\nСписок предметов ({len(self.items)} шт.):")
        for i, item in enumerate(self.items, 1):
            print(f"{i}. {item}")
    
    def solve_knapsack(self):
        """Решение задачи о рюкзаке"""
        if not self.items:
            print("Список предметов пуст")
            return
        
        if self.max_weight <= 0:
            print("Максимальный вес не установлен")
            return
        
        print("\nВыберите метод решения:")
        print("1. Рекурсивный метод")
        print("2. Метод динамического программирования")
        print("3. Жадный алгоритм (максимальный вес)")
        print("4. Жадный алгоритм (максимальное соотношение цена/вес)")
        
        try:
            choice = int(input("Ваш выбор (1-4): "))
            
            solver = KnapsackSolver(self.items, self.max_weight)
            start_time = time.time()
            
            if choice == 1:
                print("\n--- Рекурсивный метод ---")
                knapsack, calls = solver.recursive_solution()
                print(f"Количество рекурсивных вызовов: {calls}")
                
            elif choice == 2:
                print("\n--- Метод динамического программирования ---")
                knapsack = solver.dynamic_programming_solution()
                
            elif choice == 3:
                print("\n--- Жадный алгоритм (максимальный вес) ---")
                knapsack = solver.greedy_solution_by_weight()
                
            elif choice == 4:
                print("\n--- Жадный алгоритм (цена/вес) ---")
                knapsack = solver.greedy_solution_by_value_per_weight()
                
            else:
                print("Неверный выбор")
                return
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # в миллисекундах
            
            self.knapsack = knapsack
            print(f"Время выполнения: {execution_time:.2f} мс")
            print(knapsack)
            
        except ValueError:
            print("Ошибка ввода данных")
    
    def compare_methods(self):
        """Сравнение всех методов решения"""
        if not self.items:
            print("Список предметов пуст")
            return
        
        if self.max_weight <= 0:
            print("Максимальный вес не установлен")
            return
        
        print("\n" + "="*60)
        print("СРАВНЕНИЕ МЕТОДОВ РЕШЕНИЯ")
        print("="*60)
        
        solver = KnapsackSolver(self.items, self.max_weight)
        methods = [
            ("Рекурсивный", solver.recursive_solution),
            ("Динамическое программирование", solver.dynamic_programming_solution),
            ("Жадный (макс. вес)", solver.greedy_solution_by_weight),
            ("Жадный (цена/вес)", solver.greedy_solution_by_value_per_weight)
        ]
        
        results = []
        
        for method_name, method_func in methods:
            start_time = time.time()
            
            if method_name == "Рекурсивный":
                knapsack, calls = method_func()
                extra_info = f", вызовов: {calls}"
            else:
                knapsack = method_func()
                extra_info = ""
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            
            results.append({
                'method': method_name,
                'knapsack': knapsack,
                'time': execution_time,
                'extra_info': extra_info
            })
        
        # Вывод результатов
        print(f"{'Метод':<30} {'Время (мс)':<12} {'Ценность':<10} {'Вес':<8}")
        print("-" * 60)
        
        for result in results:
            k = result['knapsack']
            print(f"{result['method']:<30} {result['time']:<12.2f} {k.total_value:<10.1f} {k.current_weight:<8.1f}{result['extra_info']}")
        
        # Находим лучший метод по ценности
        best_result = max(results, key=lambda x: x['knapsack'].total_value)
        print(f"\nЛучший результат: {best_result['method']} (ценность: {best_result['knapsack'].total_value})")

def create_sample_file():
    """Создание примерного файла с данными"""
    sample_data = {
        "max_weight": 50,
        "items": [
            {"name": "Ноутбук", "weight": 10, "value": 500},
            {"name": "Книга", "weight": 2, "value": 20},
            {"name": "Камера", "weight": 5, "value": 300},
            {"name": "Еда", "weight": 8, "value": 50},
            {"name": "Вода", "weight": 3, "value": 15},
            {"name": "Одежда", "weight": 6, "value": 80},
            {"name": "Аптечка", "weight": 4, "value": 60},
            {"name": "Палатка", "weight": 15, "value": 200}
        ]
    }
    
    with open('sample_items.json', 'w', encoding='utf-8') as file:
        json.dump(sample_data, file, ensure_ascii=False, indent=2)
    
    print("Создан примерный файл 'sample_items.json'")

def main():
    manager = KnapsackManager()
    
    # Создаем примерный файл при первом запуске
    create_sample_file()
    
    while True:
        print("\n" + "="*70)
        print("Лабораторная работа №3. Задача о заполнении рюкзака")
        print("="*70)
        print("1. Загрузить предметы из файла")
        print("2. Добавить предмет")
        print("3. Изменить предмет")
        print("4. Удалить предмет")
        print("5. Задать максимальный вес рюкзака")
        print("6. Просмотреть содержимое рюкзака")
        print("7. Выбрать способ решения задачи")
        print("8. Сравнить способы решения")
        print("9. Показать все предметы")
        print("10. Сохранить предметы в файл")
        print("11. Выход")
        print("-"*70)
        
        choice = input("Выберите пункт меню (1-11): ").strip()
        
        if choice == '1':
            filename = input("Введите имя файла (по умолчанию: sample_items.json): ").strip()
            if not filename:
                filename = 'sample_items.json'
            manager.load_items_from_file(filename)
        
        elif choice == '2':
            manager.add_item()
        
        elif choice == '3':
            manager.edit_item()
        
        elif choice == '4':
            manager.delete_item()
        
        elif choice == '5':
            manager.set_max_weight()
        
        elif choice == '6':
            manager.show_knapsack()
        
        elif choice == '7':
            manager.solve_knapsack()
        
        elif choice == '8':
            manager.compare_methods()
        
        elif choice == '9':
            manager.show_items()
        
        elif choice == '10':
            filename = input("Введите имя файла для сохранения: ").strip()
            if filename:
                manager.save_items_to_file(filename)
            else:
                print("Имя файла не может быть пустым")
        
        elif choice == '11':
            print("Выход из программы.")
            break
        
        else:
            print("Неверный выбор. Пожалуйста, выберите пункт от 1 до 11.")

if __name__ == "__main__":
    main()