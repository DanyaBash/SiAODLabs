import math
import heapq
import os
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class HuffmanNode:
    """Узел дерева Хаффмана"""
    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanEncoder:
    """Класс для кодирования Хаффмана"""
    
    def __init__(self):
        self.text = ""
        self.char_freq = Counter()
        self.fixed_length_codes = {}
        self.huffman_codes = {}
        self.reverse_huffman_codes = {}
    
    def load_text_from_file(self, filename: str):
        """Загрузка текста из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.text = file.read()
            self._calculate_frequencies()
            print(f"Файл '{filename}' загружен. Символов: {len(self.text)}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False
    
    def _calculate_frequencies(self):
        """Вычисление частот символов"""
        self.char_freq = Counter(self.text)
    
    def get_sorted_frequencies(self) -> List[Tuple[str, int]]:
        """Получить отсортированный по частоте список символов"""
        return sorted(self.char_freq.items(), key=lambda x: x[1], reverse=True)
    
    def generate_fixed_length_codes(self):
        """Генерация кодов фиксированной длины"""
        chars = sorted(self.char_freq.keys())
        n = len(chars)
        if n == 0:
            return
        
        # Вычисляем минимальную длину кода
        code_length = len(bin(n - 1)) - 2
        
        self.fixed_length_codes = {}
        for i, char in enumerate(chars):
            # Преобразуем номер в двоичный код фиксированной длины
            binary_code = bin(i)[2:].zfill(code_length)
            self.fixed_length_codes[char] = binary_code
        
        return self.fixed_length_codes
    
    def generate_huffman_codes(self):
        """Генерация кодов Хаффмана"""
        if not self.char_freq:
            return {}
        
        # Создаем приоритетную очередь
        heap = []
        for char, freq in self.char_freq.items():
            node = HuffmanNode(char, freq)
            heapq.heappush(heap, node)
        
        # Строим дерево Хаффмана
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        # Генерируем коды из дерева
        root = heap[0]
        self.huffman_codes = {}
        self._generate_codes_from_tree(root, "")
        
        # Создаем обратный словарь для декодирования
        self.reverse_huffman_codes = {v: k for k, v in self.huffman_codes.items()}
        
        return self.huffman_codes
    
    def _generate_codes_from_tree(self, node: HuffmanNode, code: str):
        """Рекурсивная генерация кодов из дерева Хаффмана"""
        if node is None:
            return
        
        if node.char is not None:
            self.huffman_codes[node.char] = code
            return
        
        self._generate_codes_from_tree(node.left, code + "0")
        self._generate_codes_from_tree(node.right, code + "1")
    
    def encode_with_fixed_length(self, text: str = None) -> str:
        """Кодирование текста кодами фиксированной длины"""
        if text is None:
            text = self.text
        
        if not self.fixed_length_codes:
            self.generate_fixed_length_codes()
        
        encoded_bits = ""
        for char in text:
            if char in self.fixed_length_codes:
                encoded_bits += self.fixed_length_codes[char]
            else:
                print(f"Предупреждение: символ '{char}' не найден в алфавите")
        
        return encoded_bits
    
    def encode_with_huffman(self, text: str = None) -> str:
        """Кодирование текста кодами Хаффмана"""
        if text is None:
            text = self.text
        
        if not self.huffman_codes:
            self.generate_huffman_codes()
        
        encoded_bits = ""
        for char in text:
            if char in self.huffman_codes:
                encoded_bits += self.huffman_codes[char]
            else:
                print(f"Предупреждение: символ '{char}' не найден в алфавите")
        
        return encoded_bits
    
    def save_encoded_to_file(self, encoded_bits: str, filename: str):
        """Сохранение закодированных данных в файл"""
        try:
            # Преобразуем битовую строку в байты
            padding = 8 - len(encoded_bits) % 8
            if padding != 8:
                encoded_bits += '0' * padding
            
            # Конвертируем в байты
            byte_array = bytearray()
            for i in range(0, len(encoded_bits), 8):
                byte = encoded_bits[i:i+8]
                byte_array.append(int(byte, 2))
            
            # Сохраняем метаданные и данные
            metadata = {
                'padding': padding,
                'encoding': 'fixed' if 'fixed' in filename else 'huffman'
            }
            
            with open(filename, 'wb') as file:
                # Записываем метаданные как JSON + разделитель
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                file.write(len(metadata_bytes).to_bytes(4, 'big'))
                file.write(metadata_bytes)
                file.write(byte_array)
            
            print(f"Закодированные данные сохранены в '{filename}'")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения файла: {e}")
            return False
    
    def calculate_compression_stats(self, original_file: str, compressed_file: str):
        """Расчет статистики сжатия"""
        try:
            original_size = os.path.getsize(original_file)
            compressed_size = os.path.getsize(compressed_file)
            
            compression_ratio = (1 - compressed_size / original_size) * 100
            bits_per_char_original = original_size * 8 / len(self.text) if self.text else 0
            
            print(f"\nСтатистика сжатия:")
            print(f"Размер исходного файла: {original_size} байт")
            print(f"Размер сжатого файла: {compressed_size} байт")
            print(f"Коэффициент сжатия: {compression_ratio:.2f}%")
            print(f"Бит на символ (оригинал): {bits_per_char_original:.2f}")
            
            if self.huffman_codes:
                avg_huffman_length = sum(len(code) * self.char_freq[char] 
                                       for char, code in self.huffman_codes.items()) / len(self.text)
                print(f"Средняя длина кода Хаффмана: {avg_huffman_length:.2f} бит")
            
            return compression_ratio
            
        except Exception as e:
            print(f"Ошибка расчета статистики: {e}")
            return 0

class HuffmanMenu:
    """Класс меню для работы с программой"""
    
    def __init__(self):
        self.encoder = HuffmanEncoder()
        self.current_file = ""
    
    def display_menu(self):
        """Отображение главного меню"""
        print("\n" + "="*70)
        print("Лабораторная работа №4. Коды Хаффмана")
        print("="*70)
        print("1. Открыть текстовый файл")
        print("2. Вывести содержимое текстового файла")
        print("3. Вывести символы алфавита с частотами")
        print("4. Сгенерировать коды для символов")
        print("5. Сжать с помощью кодов фиксированной длины")
        print("6. Сжать с помощью кодов Хаффмана")
        print("7. Сравнить размеры файлов")
        print("8. Выход")
        print("-"*70)
    
    def open_file(self):
        """Открытие текстового файла"""
        filename = input("Введите имя файла: ").strip()
        if not filename:
            print("Имя файла не может быть пустым")
            return
        
        if self.encoder.load_text_from_file(filename):
            self.current_file = filename
    
    def display_file_content(self):
        """Вывод содержимого файла"""
        if not self.encoder.text:
            print("Файл не загружен")
            return
        
        print(f"\nСодержимое файла '{self.current_file}':")
        print("-" * 50)
        print(self.encoder.text[:1000] + "..." if len(self.encoder.text) > 1000 else self.encoder.text)
        print("-" * 50)
        print(f"Всего символов: {len(self.encoder.text)}")
    
    def display_frequencies(self):
        """Вывод частот символов"""
        if not self.encoder.char_freq:
            print("Файл не загружен или частоты не рассчитаны")
            return
        
        sorted_freq = self.encoder.get_sorted_frequencies()
        
        print(f"\nСимволы алфавита (всего {len(sorted_freq)} уникальных символов):")
        print("-" * 60)
        print(f"{'Символ':<10} {'Частота':<10} {'Вероятность':<12} {'Код Unicode':<15}")
        print("-" * 60)
        
        total_chars = len(self.encoder.text)
        for char, freq in sorted_freq:
            prob = freq / total_chars
            unicode_code = ord(char)
            char_display = repr(char)[1:-1]  # Экранирование специальных символов
            
            print(f"{char_display:<10} {freq:<10} {prob:<12.6f} {unicode_code:<15}")
    
    def generate_codes_menu(self):
        """Меню генерации кодов"""
        if not self.encoder.text:
            print("Файл не загружен")
            return
        
        print("\nГенерация кодов:")
        print("1. Коды фиксированной длины")
        print("2. Коды Хаффмана")
        print("3. Оба типа кодов")
        
        choice = input("Выберите опцию (1-3): ").strip()
        
        if choice in ['1', '3']:
            self.encoder.generate_fixed_length_codes()
            self._display_fixed_length_codes()
        
        if choice in ['2', '3']:
            self.encoder.generate_huffman_codes()
            self._display_huffman_codes()
    
    def _display_fixed_length_codes(self):
        """Вывод кодов фиксированной длины"""
        if not self.encoder.fixed_length_codes:
            print("Коды фиксированной длины не сгенерированы")
            return
        
        print(f"\nКоды фиксированной длины:")
        print("-" * 50)
        print(f"{'Символ':<10} {'Код':<15} {'Длина':<10}")
        print("-" * 50)
        
        for char, code in sorted(self.encoder.fixed_length_codes.items()):
            char_display = repr(char)[1:-1]
            print(f"{char_display:<10} {code:<15} {len(code):<10}")
    
    def _display_huffman_codes(self):
        """Вывод кодов Хаффмана"""
        if not self.encoder.huffman_codes:
            print("Коды Хаффмана не сгенерированы")
            return
        
        # Сортируем по длине кода
        sorted_codes = sorted(self.encoder.huffman_codes.items(), 
                            key=lambda x: (len(x[1]), x[1]))
        
        print(f"\nКоды Хаффмана:")
        print("-" * 60)
        print(f"{'Символ':<10} {'Код':<20} {'Длина':<10} {'Частота':<10}")
        print("-" * 60)
        
        for char, code in sorted_codes:
            char_display = repr(char)[1:-1]
            freq = self.encoder.char_freq[char]
            print(f"{char_display:<10} {code:<20} {len(code):<10} {freq:<10}")
        
        # Расчет энтропии и средней длины кода
        total_chars = len(self.encoder.text)
        entropy = 0
        avg_length = 0
        
        for char, freq in self.encoder.char_freq.items():
            prob = freq / total_chars
            if prob > 0:
                entropy -= prob * math.log2(prob)
            if char in self.encoder.huffman_codes:
                avg_length += prob * len(self.encoder.huffman_codes[char])
        
        print(f"\nЭнтропия: {entropy:.4f} бит/символ")
        print(f"Средняя длина кода: {avg_length:.4f} бит/символ")
        if avg_length > 0:
            print(f"Эффективность кодирования: {entropy/avg_length*100:.2f}%")
        else:
            print(f"Эффективность кодирования: 0%")
    
    def compress_fixed_length(self):
        """Сжатие с кодами фиксированной длины"""
        if not self.encoder.text:
            print("Файл не загружен")
            return
        
        if not self.encoder.fixed_length_codes:
            self.encoder.generate_fixed_length_codes()
        
        output_file = input("Введите имя выходного файла: ").strip()
        if not output_file:
            output_file = self.current_file + ".fixed.bin"
        
        encoded_bits = self.encoder.encode_with_fixed_length()
        success = self.encoder.save_encoded_to_file(encoded_bits, output_file)
        
        # Показать статистику
        if success and self.current_file:
            self.encoder.calculate_compression_stats(self.current_file, output_file)
    
    def compress_huffman(self):
        """Сжатие с кодами Хаффмана"""
        if not self.encoder.text:
            print("Файл не загружен")
            return
        
        if not self.encoder.huffman_codes:
            self.encoder.generate_huffman_codes()
        
        output_file = input("Введите имя выходного файла: ").strip()
        if not output_file:
            output_file = self.current_file + ".huffman.bin"
        
        encoded_bits = self.encoder.encode_with_huffman()
        success = self.encoder.save_encoded_to_file(encoded_bits, output_file)
        
        # Показать статистику
        if success and self.current_file:
            self.encoder.calculate_compression_stats(self.current_file, output_file)
    
    def compare_file_sizes(self):
        """Сравнение размеров файлов"""
        if not self.current_file:
            print("Файл не загружен")
            return
        
        # Определяем имена файлов для сравнения
        fixed_file = self.current_file + ".fixed.bin"
        huffman_file = self.current_file + ".huffman.bin"
        
        files_to_check = [
            (self.current_file, "Исходный файл"),
            (fixed_file, "Фиксированная длина"),
            (huffman_file, "Хаффман")
        ]
        
        # Проверяем, какие файлы существуют
        existing_files = []
        missing_files = []
        
        for filename, description in files_to_check:
            if os.path.exists(filename):
                existing_files.append((filename, description))
            else:
                missing_files.append((filename, description))
        
        # Если нет сжатых файлов, предлагаем создать их
        if len(existing_files) <= 1:  # Только исходный файл
            print("\nСжатые файлы не найдены. Хотите создать их?")
            create = input("Создать файлы для сравнения? (y/n): ").strip().lower()
            if create == 'y':
                if not self.encoder.fixed_length_codes:
                    self.encoder.generate_fixed_length_codes()
                if not self.encoder.huffman_codes:
                    self.encoder.generate_huffman_codes()
                
                # Создаем файлы
                print("Создание файлов для сравнения...")
                encoded_fixed = self.encoder.encode_with_fixed_length()
                self.encoder.save_encoded_to_file(encoded_fixed, fixed_file)
                
                encoded_huffman = self.encoder.encode_with_huffman()
                self.encoder.save_encoded_to_file(encoded_huffman, huffman_file)
                
                # Обновляем список файлов
                existing_files = [
                    (self.current_file, "Исходный файл"),
                    (fixed_file, "Фиксированная длина"),
                    (huffman_file, "Хаффман")
                ]
            else:
                return
        
        print(f"\nСравнение размеров файлов:")
        print("-" * 70)
        print(f"{'Тип файла':<25} {'Размер (байт)':<15} {'Относительно исходного':<20} {'Бит/символ':<15}")
        print("-" * 70)
        
        original_size = os.path.getsize(self.current_file)
        total_chars = len(self.encoder.text)
        
        for filename, description in existing_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                ratio = (size / original_size * 100) if original_size > 0 else 0
                bits_per_char = (size * 8 / total_chars) if total_chars > 0 else 0
                
                print(f"{description:<25} {size:<15} {ratio:.2f}%{'':<15} {bits_per_char:.2f}")
            else:
                print(f"{description:<25} {'Файл не найден':<15} {'-':<20} {'-':<15}")
        
        # Показываем недостающие файлы
        if missing_files:
            print(f"\nНе найдены файлы:")
            for filename, description in missing_files:
                print(f"  - {description}: {filename}")
        
        # Дополнительная статистика
        if total_chars > 0:
            print(f"\nДополнительная информация:")
            print(f"Количество символов: {total_chars}")
            print(f"Уникальных символов: {len(self.encoder.char_freq)}")
            
            if self.encoder.huffman_codes:
                avg_huffman_length = sum(len(code) * self.encoder.char_freq[char] 
                                       for char, code in self.encoder.huffman_codes.items()) / total_chars
                print(f"Средняя длина кода Хаффмана: {avg_huffman_length:.2f} бит")
            
            if self.encoder.fixed_length_codes:
                fixed_length = len(next(iter(self.encoder.fixed_length_codes.values())))
                print(f"Длина фиксированного кода: {fixed_length} бит")

def create_sample_file():
    """Создание примерного файла для тестирования"""
    sample_text = """В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!
Ёжик в тумане вышел погулять. Съел булку с чаем и пошел дальше.
The quick brown fox jumps over the lazy dog.
1234567890 !@#$%^&*()_+-=[]{}|;:,.<>?
"""
    
    with open('sample_text.txt', 'w', encoding='utf-8') as file:
        file.write(sample_text)
    
    print("Создан примерный файл 'sample_text.txt'")

def main():
    # Создаем примерный файл при первом запуске
    if not os.path.exists('sample_text.txt'):
        create_sample_file()
    
    menu = HuffmanMenu()
    
    while True:
        menu.display_menu()
        choice = input("Выберите пункт меню (1-8): ").strip()
        
        if choice == '1':
            menu.open_file()
        elif choice == '2':
            menu.display_file_content()
        elif choice == '3':
            menu.display_frequencies()
        elif choice == '4':
            menu.generate_codes_menu()
        elif choice == '5':
            menu.compress_fixed_length()
        elif choice == '6':
            menu.compress_huffman()
        elif choice == '7':
            menu.compare_file_sizes()
        elif choice == '8':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Пожалуйста, выберите пункт от 1 до 8.")

if __name__ == "__main__":
    main()