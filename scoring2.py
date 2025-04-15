# scoring.py
"""
Логика подсчета очков, роялти, проверки фолов и условий Фантазии
для OFC Pineapple согласно предоставленным правилам.
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import traceback # Для отладки

# --- ИМПОРТЫ ---
# Импортируем корневой Card и его константы/утилиты
# Убедимся, что импортируем все необходимое
from card import Card, card_to_str, RANK_MAP, STR_RANKS, Card as CardUtils # Используем CardUtils для статических методов

# Импортируем наши новые эвалуаторы
try:
    from src.evaluator.ofc_3card_evaluator import evaluate_3_card_ofc
    from src.evaluator.ofc_5card_evaluator import Evaluator as Evaluator5Card
    from src.evaluator.ofc_5card_lookup import LookupTable as LookupTable5Card # Импортируем константы рангов
except ImportError as e:
     print(f"ОШИБКА: Не удалось импортировать кастомные эвалуаторы в scoring.py: {e}")
     print("Убедитесь, что структура папок верна и файлы __init__.py присутствуют в src/ и src/evaluator/.")
     # Заглушки, чтобы код не падал сразу
     def evaluate_3_card_ofc(*args): return (9999, "Error", "Error")
     class Evaluator5Card:
         def evaluate(self, cards): return 9999
         def get_rank_class(self, hr): return 9
         def class_to_string(self, ci): return "Error"
     class LookupTable5Card:
         MAX_HIGH_CARD = 7462; MAX_PAIR = 6185; MAX_TWO_PAIR = 3325; MAX_THREE_OF_A_KIND = 2467
         MAX_STRAIGHT = 1609; MAX_FLUSH = 1599; MAX_FULL_HOUSE = 322; MAX_FOUR_OF_A_KIND = 166
         MAX_STRAIGHT_FLUSH = 10; MAX_ROYAL_FLUSH = 1
         RANK_CLASS_TO_STRING = {1:"SF", 2:"4K", 3:"FH", 4:"Fl", 5:"St", 6:"3K", 7:"2P", 8:"1P", 9:"HC"}
         MAX_TO_RANK_CLASS = {v: k for k, v in RANK_CLASS_TO_STRING.items()}


# --- Создаем экземпляр 5-карточного эвалуатора ---
evaluator_5card = Evaluator5Card()

# --- Константы рангов из 5-карточного эвалуатора ---
RANK_CLASS_ROYAL_FLUSH = LookupTable5Card.MAX_ROYAL_FLUSH
RANK_CLASS_STRAIGHT_FLUSH = LookupTable5Card.MAX_STRAIGHT_FLUSH
RANK_CLASS_QUADS = LookupTable5Card.MAX_FOUR_OF_A_KIND
RANK_CLASS_FULL_HOUSE = LookupTable5Card.MAX_FULL_HOUSE
RANK_CLASS_FLUSH = LookupTable5Card.MAX_FLUSH
RANK_CLASS_STRAIGHT = LookupTable5Card.MAX_STRAIGHT
RANK_CLASS_TRIPS = LookupTable5Card.MAX_THREE_OF_A_KIND
RANK_CLASS_TWO_PAIR = LookupTable5Card.MAX_TWO_PAIR
RANK_CLASS_PAIR = LookupTable5Card.MAX_PAIR
RANK_CLASS_HIGH_CARD = LookupTable5Card.MAX_HIGH_CARD

# --- Таблицы Роялти (Американские правила) ---
# Названия должны совпадать с возвращаемыми class_to_string из Evaluator5Card
# Используем LookupTable5Card.RANK_CLASS_TO_STRING для ключей, если возможно, или строки
ROYALTY_BOTTOM_POINTS = {
    "Straight": 2, "Flush": 4, "Full House": 6, "Four of a Kind": 10,
    "Straight Flush": 15, # "Royal Flush" (ранг 1) будет определен как Straight Flush
}
ROYALTY_MIDDLE_POINTS = {
    "Three of a Kind": 2, "Straight": 4, "Flush": 8, "Full House": 12,
    "Four of a Kind": 20, "Straight Flush": 30,
}
# Добавляем отдельные записи для Роял Флеша, если хотим отдельные очки
ROYALTY_BOTTOM_POINTS_RF = 25
ROYALTY_MIDDLE_POINTS_RF = 50

# Используем RANK_MAP для ключей (0='2', ..., 12='A')
ROYALTY_TOP_PAIRS = { RANK_MAP['6']: 1, RANK_MAP['7']: 2, RANK_MAP['8']: 3, RANK_MAP['9']: 4, RANK_MAP['T']: 5, RANK_MAP['J']: 6, RANK_MAP['Q']: 7, RANK_MAP['K']: 8, RANK_MAP['A']: 9 } # 66..AA
ROYALTY_TOP_TRIPS = { RANK_MAP['2']: 10, RANK_MAP['3']: 11, RANK_MAP['4']: 12, RANK_MAP['5']: 13, RANK_MAP['6']: 14, RANK_MAP['7']: 15, RANK_MAP['8']: 16, RANK_MAP['9']: 17, RANK_MAP['T']: 18, RANK_MAP['J']: 19, RANK_MAP['Q']: 20, RANK_MAP['K']: 21, RANK_MAP['A']: 22 } # 222..AAA

# --- НОВАЯ функция для получения ранга (унифицированная) ---
def get_hand_rank_safe(cards: List[Optional[int]]) -> int: # Принимает список int или None
    """
    Вызывает соответствующий эвалуатор (3 или 5 карт) и возвращает числовой ранг.
    Меньший ранг соответствует более сильной руке.
    Возвращает очень плохой ранг при ошибке/недостатке карт.
    """
    valid_cards = [c for c in cards if c is not None and isinstance(c, int)] # Фильтруем None и не-int
    num_valid = len(valid_cards)

    # Определяем ожидаемое количество карт по длине исходного списка (с None)
    expected_len = len(cards)

    if expected_len == 3:
        if num_valid != 3:
            # Возвращаем худший возможный ранг для 3 карт + штраф
            return 455 + 10 + (3 - num_valid) # 455 - худший ранг в lookup
        try:
            # evaluate_3_card_ofc ожидает карты (int)
            rank, _, _ = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
            return rank
        except Exception as e:
            print(f"Error evaluating 3-card hand { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return 455 + 100 # Худший ранг + большой штраф
    elif expected_len == 5:
        if num_valid != 5:
            # Возвращаем худший возможный ранг для 5 карт + штраф
            return RANK_CLASS_HIGH_CARD + 10 + (5 - num_valid)
        try:
            # evaluator_5card.evaluate ожидает список int
            rank = evaluator_5card.evaluate(valid_cards)
            return rank
        except Exception as e:
            print(f"Error evaluating 5-card hand { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return RANK_CLASS_HIGH_CARD + 100 # Худший ранг + большой штраф
    else:
        # Некорректное количество карт
        print(f"Warning: get_hand_rank_safe called with {expected_len} cards.")
        return RANK_CLASS_HIGH_CARD + 200 # Очень плохой ранг

# --- ОБНОВЛЕННАЯ функция для роялти ---
def get_row_royalty(cards: List[Optional[int]], row_name: str) -> int: # Принимает список int или None
    """Считает роялти для одного ряда, используя новые эвалуаторы."""
    valid_cards = [c for c in cards if c is not None and isinstance(c, int)]
    num_cards = len(valid_cards)
    royalty = 0

    if row_name == "top":
        if num_cards != 3: return 0
        try:
            # Используем 3-карточный эвалуатор
            _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])

            if type_str == 'Trips':
                # Извлекаем ранг из строки типа 'AAA'
                rank_char = rank_str[0]
                rank_index = RANK_MAP.get(rank_char)
                if rank_index is not None:
                    royalty = ROYALTY_TOP_TRIPS.get(rank_index, 0)
            elif type_str == 'Pair':
                # Извлекаем ранг пары из строки типа 'QQK' -> 'Q'
                pair_rank_char = rank_str[0] # У пары всегда две одинаковые карты в начале строки
                # Проверяем, что это действительно пара (второй символ такой же)
                if len(rank_str) == 3 and rank_str[1] == pair_rank_char:
                     rank_index = RANK_MAP.get(pair_rank_char)
                     if rank_index is not None:
                         royalty = ROYALTY_TOP_PAIRS.get(rank_index, 0)
            return royalty
        except Exception as e:
            print(f"Error calculating top row royalty for { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return 0

    elif row_name in ["middle", "bottom"]:
        if num_cards != 5: return 0
        try:
            # Используем 5-карточный эвалуатор
            rank_eval = get_hand_rank_safe(valid_cards) # Получаем числовой ранг
            rank_class = evaluator_5card.get_rank_class(rank_eval) # Получаем класс руки (int)
            hand_name = evaluator_5card.class_to_string(rank_class) # Получаем название руки (str)

            # Особый случай для Роял Флеша (ранг 1)
            is_royal = (rank_eval == 1)

            table = ROYALTY_MIDDLE_POINTS if row_name == "middle" else ROYALTY_BOTTOM_POINTS

            # Ищем роялти по названию руки
            royalty = table.get(hand_name, 0)

            # Добавляем бонус за Роял Флеш, если он есть в таблице
            if is_royal:
                 if row_name == "middle": royalty = max(royalty, ROYALTY_MIDDLE_POINTS_RF)
                 elif row_name == "bottom": royalty = max(royalty, ROYALTY_BOTTOM_POINTS_RF)

            # Корректировка для среднего ряда: Трипс учитывается только там
            if row_name == "bottom" and hand_name == "Three of a Kind":
                 royalty = 0

            return royalty
        except Exception as e:
            print(f"Error calculating {row_name} row royalty for { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return 0
    else:
        return 0

# --- Функция проверки фола (остается без изменений, т.к. использует get_hand_rank_safe) ---
def check_board_foul(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    """Проверяет фол доски (только для полных досок)."""
    # Проверяем, что все ряды полностью заполнены
    if sum(1 for c in top if c is not None) != 3 or \
       sum(1 for c in middle if c is not None) != 5 or \
       sum(1 for c in bottom if c is not None) != 5:
        return False # Не фол, если доска не полная

    rank_t = get_hand_rank_safe(top)
    rank_m = get_hand_rank_safe(middle)
    rank_b = get_hand_rank_safe(bottom)

    # Фол, если нижний ряд слабее среднего ИЛИ средний ряд слабее верхнего
    # Меньший ранг означает более сильную руку
    is_foul = not (rank_b <= rank_m <= rank_t)
    return is_foul


# --- ОБНОВЛЕННАЯ функция для входа в Fantasyland ---
def get_fantasyland_entry_cards(top: List[Optional[int]]) -> int:
    """Возвращает кол-во карт для ФЛ при входе (0 если нет квалификации)."""
    valid_cards = [c for c in top if c is not None and isinstance(c, int)]
    if len(valid_cards) != 3: return 0
    try:
        # Используем 3-карточный эвалуатор
        _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])

        if type_str == 'Trips':
            # Любой трипс дает 17 карт
            return 17
        elif type_str == 'Pair':
            # Извлекаем ранг пары
            pair_rank_char = rank_str[0]
            if pair_rank_char == 'Q': return 14
            if pair_rank_char == 'K': return 15
            if pair_rank_char == 'A': return 16
        return 0 # Не пара QQ+ и не трипс
    except Exception as e:
        print(f"Error checking Fantasyland entry for { [card_to_str(c) for c in valid_cards] }: {e}")
        traceback.print_exc()
        return 0

# --- ОБНОВЛЕННАЯ функция для удержания Fantasyland ---
def check_fantasyland_stay(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    """Проверяет условия удержания ФЛ (Сет+ топ ИЛИ Каре+ боттом)."""
    valid_top = [c for c in top if c is not None and isinstance(c, int)]
    valid_middle = [c for c in middle if c is not None and isinstance(c, int)] # Не используется, но для полноты
    valid_bottom = [c for c in bottom if c is not None and isinstance(c, int)]

    if len(valid_top) != 3 or len(valid_middle) != 5 or len(valid_bottom) != 5:
        return False

    # 1. Проверяем Сет на топе
    try:
        _, type_str_top, _ = evaluate_3_card_ofc(valid_top[0], valid_top[1], valid_top[2])
        if type_str_top == 'Trips':
            return True
    except Exception as e:
        print(f"Error checking top for Fantasyland stay { [card_to_str(c) for c in valid_top] }: {e}")
        traceback.print_exc()
        # Продолжаем проверку боттома

    # 2. Проверяем Каре или лучше на боттоме
    try:
        rank_b = get_hand_rank_safe(valid_bottom)
        # Каре (166) или лучше (меньше ранг)
        if rank_b <= RANK_CLASS_QUADS:
            return True
    except Exception as e:
        print(f"Error checking bottom for Fantasyland stay { [card_to_str(c) for c in valid_bottom] }: {e}")
        traceback.print_exc()

    return False

# --- Функция подсчета очков (зависит от PlayerBoard, который использует _get_rank) ---
# Оставляем без изменений, так как она использует методы PlayerBoard,
# которые в свою очередь будут использовать обновленные get_hand_rank_safe и get_row_royalty
def calculate_headsup_score(board1: 'PlayerBoard', board2: 'PlayerBoard') -> int:
    """Считает очки между двумя игроками (с точки зрения Игрока 1)."""
    # board1 и board2 теперь PlayerBoard
    # Проверка фола и расчет роялти теперь внутри PlayerBoard используют обновленные функции
    foul1 = board1.is_complete() and board1.check_and_set_foul()
    foul2 = board2.is_complete() and board2.check_and_set_foul()

    # Важно: Получаем роялти ПОСЛЕ проверки фола, т.к. фол обнуляет роялти
    r1 = board1.get_total_royalty() # get_total_royalty вызовет get_royalties, который учтет foul1
    r2 = board2.get_total_royalty() # get_total_royalty вызовет get_royalties, который учтет foul2

    if foul1 and foul2: return 0
    if foul1: return -(6 + r2) # Игрок 1 фол, проигрывает 6 очков + роялти Игрока 2
    if foul2: return 6 + r1   # Игрок 2 фол, выигрывает 6 очков + роялти Игрока 1

    # Если фолов нет, сравниваем линии
    score1 = 0
    # Используем кэшированные ранги из board._get_rank(), который вызывает get_hand_rank_safe
    rank_t1 = board1._get_rank('top')
    rank_m1 = board1._get_rank('middle')
    rank_b1 = board1._get_rank('bottom')
    rank_t2 = board2._get_rank('top')
    rank_m2 = board2._get_rank('middle')
    rank_b2 = board2._get_rank('bottom')

    wins1 = 0
    if rank_t1 < rank_t2: wins1 += 1
    elif rank_t2 < rank_t1: wins1 -= 1

    if rank_m1 < rank_m2: wins1 += 1
    elif rank_m2 < rank_m1: wins1 -= 1

    if rank_b1 < rank_b2: wins1 += 1
    elif rank_b2 < rank_b1: wins1 -= 1

    # Очки за линии
    score1 += wins1

    # Бонус за scoop (выигрыш всех 3 линий)
    if wins1 == 3: score1 += 3
    elif wins1 == -3: score1 -= 3 # Проигрыш всех 3 линий

    # Добавляем разницу роялти
    score1 += (r1 - r2)

    return score1