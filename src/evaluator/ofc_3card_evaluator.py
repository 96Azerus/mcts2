# -*- coding: utf-8 -*-
# Этот файл содержит функцию для оценки 3-карточной руки OFC
# Он ИМПОРТИРУЕТ таблицу поиска из ofc_3card_lookup.py

# Импортируем таблицу и корневой Card
from .ofc_3card_lookup import three_card_lookup
# Важно: Импортируем Card из корневой директории относительно точки запуска (app.py)
# или используем относительный путь, если структура позволяет
try:
    # Попытка импорта относительно корня проекта (если запускается из корня)
    from card import Card as RootCardUtils # Используем алиас, чтобы не конфликтовать
    from card import RANK_MAP as ROOT_RANK_MAP # Импортируем RANK_MAP
except ImportError:
    # Попытка импорта, если evaluator находится в пакете src
    # Это может потребовать __init__.py в корне и правильной настройки PYTHONPATH
    try:
         # Предполагаем, что scoring.py находится на том же уровне, что и папка src
         # Это не очень хороший способ, лучше настроить PYTHONPATH или структуру проекта
         import sys
         import os
         # Добавляем корень проекта в путь (если запускается не из корня)
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
         if project_root not in sys.path:
              sys.path.insert(0, project_root)
         from card import Card as RootCardUtils
         from card import RANK_MAP as ROOT_RANK_MAP
         print("DEBUG: Imported root card utils via sys.path modification.")
    except ImportError as e_inner:
         print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать корневой card.py в ofc_3card_evaluator: {e_inner}. Используется запасной вариант.")
         # Запасной вариант, если импорт не сработал (менее надежно)
         class RootCardUtils:
             @staticmethod
             def get_rank_int(card_int): return (card_int >> 8) & 0xF
             @staticmethod
             def int_to_str(card_int):
                  rank_int = RootCardUtils.get_rank_int(card_int)
                  ranks = '23456789TJQKA'
                  return ranks[rank_int] + 'x' if 0 <= rank_int < 13 else '??'
         ROOT_RANK_MAP = {rank: i for i, rank in enumerate('23456789TJQKA')}


def evaluate_3_card_ofc(card1: int, card2: int, card3: int) -> Tuple[int, str, str]: # Принимает int
    """
    Оценивает 3-карточную руку по правилам OFC, используя предрасчитанную таблицу.
    Карты должны быть целочисленными представлениями из корневого card.py.
    Возвращает кортеж: (rank, type_string, rank_string).
    Меньший ранг соответствует более сильной руке.
    """
    ranks = []
    for card_int in [card1, card2, card3]:
        if isinstance(card_int, int): # Если передан целочисленный Card
             try:
                 # Используем статический метод RootCardUtils для получения ранга
                 rank_int = RootCardUtils.get_rank_int(card_int)
                 if 0 <= rank_int <= 12:
                     ranks.append(rank_int)
                 else:
                      raise ValueError(f"Неверный ранг {rank_int}, извлеченный из карты int: {card_int}")
             except AttributeError:
                  # Если у RootCardUtils нет метода get_rank_int (маловероятно)
                  raise TypeError(f"Переданный int {card_int} не является валидным представлением RootCard.")
             except Exception as e: # Ловим другие возможные ошибки
                  raise ValueError(f"Ошибка при обработке карты int {card_int}: {e}")
        elif card_int is None:
             raise ValueError("Передана пустая карта (None)")
        else:
            # Убрали поддержку строк, т.к. весь проект использует int
            raise TypeError(f"Неподдерживаемый тип карты: {type(card_int)}. Ожидался int (RootCard).")

    if len(ranks) != 3:
         raise ValueError("Должно быть передано ровно 3 карты")

    # Канонический ключ - отсортированный по убыванию кортеж рангов (0-12)
    lookup_key = tuple(sorted(ranks, reverse=True))

    result = three_card_lookup.get(lookup_key)
    if result is None:
        # Этого не должно произойти, если таблица сгенерирована правильно
        raise ValueError(f"Не найдена комбинация для ключа: {lookup_key} (исходные ранги: {ranks})")

    # Возвращаем ранг (число), тип (строка), строку рангов (строка)
    return result

# Пример использования внутри модуля (для тестирования)
if __name__ == '__main__':
    # Тестируем разные руки
    # Используем строки для создания int карт через RootCardUtils.new
    try:
        hand1_int = (RootCardUtils.new('Ah'), RootCardUtils.new('Ad'), RootCardUtils.new('As')) # Трипс тузов (лучшая)
        hand2_int = (RootCardUtils.new('Qh'), RootCardUtils.new('Qs'), RootCardUtils.new('2d')) # Пара дам
        hand3_int = (RootCardUtils.new('Kd'), RootCardUtils.new('5s'), RootCardUtils.new('2h')) # Старший король
        hand4_int = (RootCardUtils.new('6c'), RootCardUtils.new('6d'), RootCardUtils.new('Ts')) # Пара шестерок
        hand5_int = (RootCardUtils.new('2h'), RootCardUtils.new('2d'), RootCardUtils.new('2s')) # Трипс двоек
        hand6_int = (RootCardUtils.new('Jd'), RootCardUtils.new('Th'), RootCardUtils.new('9s')) # Старший валет
        hand7_int = (RootCardUtils.new('5h'), RootCardUtils.new('3d'), RootCardUtils.new('2c')) # 532 старшие (худшая)

        tests = [hand1_int, hand2_int, hand3_int, hand4_int, hand5_int, hand6_int, hand7_int]
        results = []
        print("--- Тестирование evaluate_3_card_ofc ---")
        for i, hand_int in enumerate(tests):
            try:
                # Передаем int карты в функцию
                rank, type_str, rank_str = evaluate_3_card_ofc(*hand_int)
                hand_str_repr = tuple(RootCardUtils.int_to_str(c) for c in hand_int)
                print(f"Тест {i+1}: {hand_str_repr} -> Ранг: {rank}, Тип: {type_str}, Строка: {rank_str}")
                results.append(rank)
            except Exception as e:
                hand_str_repr = tuple(RootCardUtils.int_to_str(c) for c in hand_int)
                print(f"Ошибка при тесте {i+1} {hand_str_repr}: {e}")
                traceback.print_exc()

        # Проверка сортировки рангов (меньше = лучше)
        if results:
            print("\nПроверка порядка рангов (должны быть отсортированы по возрастанию):")
            is_sorted = all(results[i] <= results[i+1] for i in range(len(results)-1))
            print(f"Ранги: {results}")
            # Ожидаемые ранги из lookup: AAA(1), 222(13), QQx(40), K52(287), 66T(114), JT9(336), 532(455)
            expected_ranks = [1, 40, 287, 114, 13, 336, 455]
            expected_sorted = sorted(expected_ranks)
            print(f"Ожидаемый порядок: {expected_sorted}")
            print(f"Полученный порядок: {sorted(results)}")
            print(f"Совпадает: {sorted(results) == expected_sorted}")

        print("\nФункция оценки 3-карточных рук работает.")

    except NameError:
         print("\nНе удалось выполнить тесты: RootCardUtils не определен (ошибка импорта корневого card.py).")
    except Exception as e_main:
         print(f"\nНе удалось выполнить тесты: {e_main}")
         traceback.print_exc()
