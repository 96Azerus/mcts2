# scoring.py
"""
Логика подсчета очков, роялти, проверки фолов и условий Фантазии
для OFC Pineapple согласно предоставленным правилам.
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import traceback # Для отладки

# --- ИМПОРТЫ ---
from card import Card, card_to_str, RANK_MAP, STR_RANKS, Card as CardUtils
try:
    from src.evaluator.ofc_3card_evaluator import evaluate_3_card_ofc
    from src.evaluator.ofc_5card_evaluator import Evaluator as Evaluator5Card
    # Импортируем LookupTable только для доступа к константам MAX_...
    from src.evaluator.ofc_5card_lookup import LookupTable as LookupTable5Card
except ImportError as e:
     print(f"ОШИБКА: Не удалось импортировать кастомные эвалуаторы в scoring.py: {e}")
     # ... (заглушки как раньше) ...
     def evaluate_3_card_ofc(*args): return (9999, "Error", "Error")
     class Evaluator5Card:
         def evaluate(self, cards): return 9999
         def get_rank_class(self, hr): return 9
         def class_to_string(self, ci): return "Error"
     class LookupTable5Card:
         MAX_HIGH_CARD = 7462; MAX_PAIR = 6185; MAX_TWO_PAIR = 3325; MAX_THREE_OF_A_KIND = 2467
         MAX_STRAIGHT = 1609; MAX_FLUSH = 1599; MAX_FULL_HOUSE = 322; MAX_FOUR_OF_A_KIND = 166
         MAX_STRAIGHT_FLUSH = 10
         RANK_CLASS_TO_STRING = {1:"SF", 2:"4K", 3:"FH", 4:"Fl", 5:"St", 6:"3K", 7:"2P", 8:"1P", 9:"HC"}
         MAX_TO_RANK_CLASS = {v: k for k, v in RANK_CLASS_TO_STRING.items()}


# --- Создаем экземпляр 5-карточного эвалуатора ---
evaluator_5card = Evaluator5Card()

# --- Константы рангов из 5-карточного эвалуатора ---
# --- ИСПРАВЛЕНО: Убрана ссылка на несуществующую константу ---
RANK_CLASS_ROYAL_FLUSH = 1 # Роял Флеш всегда имеет ранг 1
RANK_CLASS_STRAIGHT_FLUSH = LookupTable5Card.MAX_STRAIGHT_FLUSH # 10
RANK_CLASS_QUADS = LookupTable5Card.MAX_FOUR_OF_A_KIND # 166
RANK_CLASS_FULL_HOUSE = LookupTable5Card.MAX_FULL_HOUSE # 322
RANK_CLASS_FLUSH = LookupTable5Card.MAX_FLUSH # 1599
RANK_CLASS_STRAIGHT = LookupTable5Card.MAX_STRAIGHT # 1609
RANK_CLASS_TRIPS = LookupTable5Card.MAX_THREE_OF_A_KIND # 2467
RANK_CLASS_TWO_PAIR = LookupTable5Card.MAX_TWO_PAIR # 3325
RANK_CLASS_PAIR = LookupTable5Card.MAX_PAIR # 6185
RANK_CLASS_HIGH_CARD = LookupTable5Card.MAX_HIGH_CARD # 7462

# --- Таблицы Роялти (Американские правила) ---
ROYALTY_BOTTOM_POINTS = {
    "Straight": 2, "Flush": 4, "Full House": 6, "Four of a Kind": 10,
    "Straight Flush": 15,
}
ROYALTY_MIDDLE_POINTS = {
    "Three of a Kind": 2, "Straight": 4, "Flush": 8, "Full House": 12,
    "Four of a Kind": 20, "Straight Flush": 30,
}
ROYALTY_BOTTOM_POINTS_RF = 25 # Отдельно для проверки rank == 1
ROYALTY_MIDDLE_POINTS_RF = 50 # Отдельно для проверки rank == 1

ROYALTY_TOP_PAIRS = { RANK_MAP['6']: 1, RANK_MAP['7']: 2, RANK_MAP['8']: 3, RANK_MAP['9']: 4, RANK_MAP['T']: 5, RANK_MAP['J']: 6, RANK_MAP['Q']: 7, RANK_MAP['K']: 8, RANK_MAP['A']: 9 }
ROYALTY_TOP_TRIPS = { RANK_MAP['2']: 10, RANK_MAP['3']: 11, RANK_MAP['4']: 12, RANK_MAP['5']: 13, RANK_MAP['6']: 14, RANK_MAP['7']: 15, RANK_MAP['8']: 16, RANK_MAP['9']: 17, RANK_MAP['T']: 18, RANK_MAP['J']: 19, RANK_MAP['Q']: 20, RANK_MAP['K']: 21, RANK_MAP['A']: 22 }

# --- Функция get_hand_rank_safe (без изменений) ---
def get_hand_rank_safe(cards: List[Optional[int]]) -> int:
    valid_cards = [c for c in cards if c is not None and isinstance(c, int)]
    num_valid = len(valid_cards)
    expected_len = len(cards)

    if expected_len == 3:
        if num_valid != 3: return 455 + 10 + (3 - num_valid)
        try:
            rank, _, _ = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
            return rank
        except Exception as e:
            print(f"Error evaluating 3-card hand { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return 455 + 100
    elif expected_len == 5:
        if num_valid != 5: return RANK_CLASS_HIGH_CARD + 10 + (5 - num_valid)
        try:
            rank = evaluator_5card.evaluate(valid_cards)
            return rank
        except Exception as e:
            print(f"Error evaluating 5-card hand { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return RANK_CLASS_HIGH_CARD + 100
    else:
        print(f"Warning: get_hand_rank_safe called with {expected_len} cards.")
        return RANK_CLASS_HIGH_CARD + 200

# --- Функция get_row_royalty (без изменений в логике, но использует исправленные константы) ---
def get_row_royalty(cards: List[Optional[int]], row_name: str) -> int:
    valid_cards = [c for c in cards if c is not None and isinstance(c, int)]
    num_cards = len(valid_cards)
    royalty = 0

    if row_name == "top":
        if num_cards != 3: return 0
        try:
            _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
            if type_str == 'Trips':
                rank_char = rank_str[0]
                rank_index = RANK_MAP.get(rank_char)
                if rank_index is not None: royalty = ROYALTY_TOP_TRIPS.get(rank_index, 0)
            elif type_str == 'Pair':
                pair_rank_char = rank_str[0]
                if len(rank_str) == 3 and rank_str[1] == pair_rank_char:
                     rank_index = RANK_MAP.get(pair_rank_char)
                     if rank_index is not None: royalty = ROYALTY_TOP_PAIRS.get(rank_index, 0)
            return royalty
        except Exception as e:
            print(f"Error calculating top row royalty for { [card_to_str(c) for c in valid_cards] }: {e}")
            traceback.print_exc()
            return 0

    elif row_name in ["middle", "bottom"]:
        if num_cards != 5: return 0
        try:
            rank_eval = get_hand_rank_safe(valid_cards)
            # --- ИСПРАВЛЕНО: Проверяем ранг 1 для Роял Флеша ---
            is_royal = (rank_eval == RANK_CLASS_ROYAL_FLUSH) # rank_eval == 1

            rank_class = evaluator_5card.get_rank_class(rank_eval)
            hand_name = evaluator_5card.class_to_string(rank_class)

            table = ROYALTY_MIDDLE_POINTS if row_name == "middle" else ROYALTY_BOTTOM_POINTS

            # Ищем роялти по названию руки (SF, 4K, FH, Fl, St, 3K)
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

# --- Остальные функции (check_board_foul, get_fantasyland_entry_cards, check_fantasyland_stay, calculate_headsup_score) остаются без изменений ---
# Они используют get_hand_rank_safe и get_row_royalty, которые уже исправлены.

def check_board_foul(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    if sum(1 for c in top if c is not None) != 3 or \
       sum(1 for c in middle if c is not None) != 5 or \
       sum(1 for c in bottom if c is not None) != 5:
        return False
    rank_t = get_hand_rank_safe(top)
    rank_m = get_hand_rank_safe(middle)
    rank_b = get_hand_rank_safe(bottom)
    is_foul = not (rank_b <= rank_m <= rank_t)
    return is_foul

def get_fantasyland_entry_cards(top: List[Optional[int]]) -> int:
    valid_cards = [c for c in top if c is not None and isinstance(c, int)]
    if len(valid_cards) != 3: return 0
    try:
        _, type_str, rank_str = evaluate_3_card_ofc(valid_cards[0], valid_cards[1], valid_cards[2])
        if type_str == 'Trips': return 17
        elif type_str == 'Pair':
            pair_rank_char = rank_str[0]
            if pair_rank_char == 'Q': return 14
            if pair_rank_char == 'K': return 15
            if pair_rank_char == 'A': return 16
        return 0
    except Exception as e:
        print(f"Error checking Fantasyland entry for { [card_to_str(c) for c in valid_cards] }: {e}")
        traceback.print_exc()
        return 0

def check_fantasyland_stay(top: List[Optional[int]], middle: List[Optional[int]], bottom: List[Optional[int]]) -> bool:
    valid_top = [c for c in top if c is not None and isinstance(c, int)]
    valid_middle = [c for c in middle if c is not None and isinstance(c, int)]
    valid_bottom = [c for c in bottom if c is not None and isinstance(c, int)]
    if len(valid_top) != 3 or len(valid_middle) != 5 or len(valid_bottom) != 5: return False

    try: # Проверка топа
        _, type_str_top, _ = evaluate_3_card_ofc(valid_top[0], valid_top[1], valid_top[2])
        if type_str_top == 'Trips': return True
    except Exception as e:
        print(f"Error checking top for Fantasyland stay { [card_to_str(c) for c in valid_top] }: {e}")
        traceback.print_exc()

    try: # Проверка боттома
        rank_b = get_hand_rank_safe(valid_bottom)
        if rank_b <= RANK_CLASS_QUADS: return True # Каре или лучше
    except Exception as e:
        print(f"Error checking bottom for Fantasyland stay { [card_to_str(c) for c in valid_bottom] }: {e}")
        traceback.print_exc()

    return False

def calculate_headsup_score(board1: 'PlayerBoard', board2: 'PlayerBoard') -> int:
    foul1 = board1.is_complete() and board1.check_and_set_foul()
    foul2 = board2.is_complete() and board2.check_and_set_foul()
    r1 = board1.get_total_royalty()
    r2 = board2.get_total_royalty()

    if foul1 and foul2: return 0
    if foul1: return -(6 + r2)
    if foul2: return 6 + r1

    score1 = 0
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

    score1 += wins1
    if wins1 == 3: score1 += 3
    elif wins1 == -3: score1 -= 3
    score1 += (r1 - r2)
    return score1
