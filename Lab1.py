class NotationConverter:
    def __init__(self):
        # Определяем приоритеты операций
        self.precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
    
    def is_operator(self, char):
        """Проверяет, является ли символ оператором"""
        return char in '+-*/^'
    
    def get_precedence(self, operator):
        """Возвращает приоритет оператора"""
        return self.precedence.get(operator, 0)
    
    def infix_to_postfix(self, expression):
        """Переводит инфиксное выражение в постфиксное (обратную польскую запись)"""
        # Удаляем пробелы и добавляем пробелы вокруг операторов и скобок для удобства разбиения
        expression = expression.replace(' ', '')
        for op in '+-*/^()':
            expression = expression.replace(op, f' {op} ')
        
        tokens = expression.split()
        output = []
        stack = []
        
        for token in tokens:
            if token.isdigit() or (token.replace('.', '').isdigit() and token.count('.') <= 1):
                # Если токен - число, добавляем в выходную очередь
                output.append(token)
            elif token == '(':
                # Если токен - открывающая скобка, помещаем в стек
                stack.append(token)
            elif token == ')':
                # Если токен - закрывающая скобка, выталкиваем операторы из стека до открывающей скобки
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack and stack[-1] == '(':
                    stack.pop()  # Удаляем открывающую скобку
                else:
                    raise ValueError("Несбалансированные скобки")
            elif self.is_operator(token):
                # Если токен - оператор
                while (stack and stack[-1] != '(' and 
                       self.get_precedence(stack[-1]) >= self.get_precedence(token)):
                    output.append(stack.pop())
                stack.append(token)
            else:
                raise ValueError(f"Некорректный символ: {token}")
        
        # Выталкиваем оставшиеся операторы из стека
        while stack:
            if stack[-1] == '(':
                raise ValueError("Несбалансированные скобки")
            output.append(stack.pop())
        
        return ' '.join(output)
    
    def evaluate_postfix(self, expression):
        """Вычисляет значение постфиксного выражения"""
        tokens = expression.split()
        stack = []
        
        for token in tokens:
            if token.isdigit() or (token.replace('.', '').isdigit() and token.count('.') <= 1):
                # Если токен - число, помещаем в стек
                stack.append(float(token))
            elif self.is_operator(token):
                # Если токен - оператор, извлекаем операнды и выполняем операцию
                if len(stack) < 2:
                    raise ValueError("Недостаточно операндов для операции")
                
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    if b == 0:
                        raise ValueError("Деление на ноль")
                    result = a / b
                elif token == '^':
                    result = a ** b
                
                stack.append(result)
            else:
                raise ValueError(f"Некорректный символ в постфиксном выражении: {token}")
        
        if len(stack) != 1:
            raise ValueError("Некорректное выражение")
        
        return stack[0]
    
    def evaluate_infix(self, expression):
        """Вычисляет значение инфиксного выражения через преобразование в постфиксное"""
        postfix = self.infix_to_postfix(expression)
        return self.evaluate_postfix(postfix)


def run_tests():
    """Запускает автоматические тесты"""
    converter = NotationConverter()
    
    test_cases = [
        # (инфикс, постфикс, результат)
        ("2 + 3", "2 3 +", 5),
        ("10 - 4", "10 4 -", 6),
        ("3 * 4", "3 4 *", 12),
        ("15 / 3", "15 3 /", 5),
        ("2 + 3 * 4", "2 3 4 * +", 14),
        ("(2 + 3) * 4", "2 3 + 4 *", 20),
        ("2 ^ 3", "2 3 ^", 8),
        ("10 + 2 * 3 - 4 / 2", "10 2 3 * + 4 2 / -", 14),
        ("(3 + 5) * 10 - 17 * 2", "3 5 + 10 * 17 2 * -", 46),
        ("5.5 + 2.5", "5.5 2.5 +", 8),
        ("2 + 3 * (4 - 1)", "2 3 4 1 - * +", 11),
    ]
    
    print("\n" + "="*70)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ ПРОГРАММЫ")
    print("="*70)
    
    all_passed = True
    passed_count = 0
    
    for i, (infix, expected_postfix, expected_result) in enumerate(test_cases, 1):
        try:
            # Тест преобразования
            actual_postfix = converter.infix_to_postfix(infix)
            postfix_ok = actual_postfix == expected_postfix
            
            # Тест вычисления
            actual_result = converter.evaluate_postfix(actual_postfix)
            result_ok = abs(actual_result - expected_result) < 0.0001
            
            status = "✓ ПРОЙДЕН" if (postfix_ok and result_ok) else "✗ НЕ ПРОЙДЕН"
            
            print(f"Тест {i:2d}: {status}")
            print(f"   Инфикс:    {infix}")
            print(f"   Постфикс:  {actual_postfix} (ожидалось: {expected_postfix})")
            print(f"   Результат: {actual_result} (ожидалось: {expected_result})")
            
            if postfix_ok and result_ok:
                passed_count += 1
            else:
                all_passed = False
                
        except Exception as e:
            print(f"Тест {i:2d}: ✗ ОШИБКА")
            print(f"   Выражение: {infix}")
            print(f"   Ошибка: {e}")
            all_passed = False
    
    print("="*70)
    print(f"РЕЗУЛЬТАТЫ: {passed_count}/{len(test_cases)} тестов пройдено")
    if all_passed:
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("✗ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
    print("="*70)


def run_error_tests():
    """Тестирование обработки ошибок"""
    converter = NotationConverter()
    
    error_cases = [
        ("(2 + 3", "Несбалансированные скобки"),
        ("2 + ) 3", "Несбалансированные скобки"),
        ("5 / 0", "Деление на ноль"),
        ("2 + a", "Некорректный символ"),
        ("2 3 +", "Некорректное выражение (для инфиксного)"),
        ("+", "Недостаточно операндов"),
    ]
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ОБРАБОТКИ ОШИБОК")
    print("="*60)
    
    for infix, expected_error in error_cases:
        try:
            result = converter.evaluate_infix(infix)
            print(f"✗ Ожидалась ошибка '{expected_error}', но получен результат: {result}")
        except Exception as e:
            print(f"✓ Корректно обработана ошибка: {e}")
    
    print("="*60)


def main():
    converter = NotationConverter()
    
    while True:
        print("\n" + "="*50)
        print("Лабораторная работа №1. Инфиксная и постфиксная нотации")
        print("="*50)
        print("1. Посчитать выражение в инфиксном виде")
        print("2. Посчитать выражение в постфиксном виде")
        print("3. Перевести выражение из инфиксного в постфиксный вид")
        print("4. Запустить автоматические тесты")
        print("5. Тестирование обработки ошибок")
        print("6. Выход")
        print("-"*50)
        
        choice = input("Выберите пункт меню (1-6): ").strip()
        
        if choice == '1':
            expression = input("Введите инфиксное выражение: ").strip()
            try:
                result = converter.evaluate_infix(expression)
                print(f"Результат: {result}")
            except Exception as e:
                print(f"Ошибка: {e}")
        
        elif choice == '2':
            expression = input("Введите постфиксное выражение: ").strip()
            try:
                result = converter.evaluate_postfix(expression)
                print(f"Результат: {result}")
            except Exception as e:
                print(f"Ошибка: {e}")
        
        elif choice == '3':
            expression = input("Введите инфиксное выражение: ").strip()
            try:
                postfix = converter.infix_to_postfix(expression)
                print(f"Постфиксная запись: {postfix}")
            except Exception as e:
                print(f"Ошибка: {e}")
        
        elif choice == '4':
            run_tests()
        
        elif choice == '5':
            run_error_tests()
        
        elif choice == '6':
            print("Выход из программы.")
            break
        
        else:
            print("Неверный выбор. Пожалуйста, выберите пункт от 1 до 6.")


def demonstrate_examples():
    """Демонстрация работы на примерах из лекции"""
    converter = NotationConverter()
    
    examples = [
        "3 + 5",
        "(3 + 5) * 10", 
        "(3 + 5) * 10 - 17 * 2"
    ]
    
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ПРИМЕРОВ ИЗ ЛЕКЦИИ")
    print("="*60)
    
    for example in examples:
        try:
            postfix = converter.infix_to_postfix(example)
            result = converter.evaluate_postfix(postfix)
            print(f"Инфикс:  {example}")
            print(f"Постфикс: {postfix}")
            print(f"Результат: {result}")
            print("-" * 40)
        except Exception as e:
            print(f"Ошибка в выражении '{example}': {e}")
            print("-" * 40)


if __name__ == "__main__":
    # Демонстрация работы программы
    demonstrate_examples()
    
    # Запуск основного меню
    main()