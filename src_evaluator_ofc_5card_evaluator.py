# -*- coding: utf-8 -*-
import itertools
# --- ИЗМЕНЕНИЕ: Импортируем корневой Card и его утилиты ---
try:
    from card import Card as CardUtils # Используем алиас
    from card import Card as RootCard # Для ясности, что это корневой
except ImportError:
    try:
         import sys
         import os
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
         if project_root not in sys.path:
              sys.path.insert(0, project_root)
         from card import Card as CardUtils
         from card import Card as RootCard
         print("DEBUG: Imported root card utils via sys.path modification in 5-card evaluator.")
    except ImportError as e_inner:
         print(f"ОШИБКА: Не удалось импортировать корневой card.py в ofc_5card_evaluator: {e_inner}. Используется запасной вариант.")
         class CardUtils: # Заглушка
             PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
             INT_RANKS = range(13)
             @staticmethod
             def get_prime(card_int): return card_int & 0x3F
             @staticmethod
             def prime_product_from_hand(card_ints):
                 p = 1
                 for c in card_ints: p *= CardUtils.get_prime(c)
                 return p
             @staticmethod
             def prime_product_from_rankbits(rankbits):
                 p = 1
                 for i in CardUtils.INT_RANKS:
                     if rankbits & (1 << i): p *= CardUtils.PRIMES[i]
                 return p
         RootCard = CardUtils # Используем ту же заглушку

# Используем относительный импорт для LookupTable внутри пакета evaluator
from .ofc_5card_lookup import LookupTable

class Evaluator(object):
    """
    Evaluates 5-card hand strengths using a variant of Cactus Kev's algorithm.
    Uses integer card representations from the root card.py.
    """

    def __init__(self):
        self.table = LookupTable()
        # Убрали _six и _seven, т.к. OFC использует только 5 карт
        self.hand_size_map = {
            5 : self._five,
        }

    def evaluate(self, cards: List[int]) -> int: # Принимает список int
        """
        This is the function that the user calls to get a hand rank for 5 cards.
        Input: List of 5 integer card representations from root card.py.
        Returns rank (int), lower is better.
        """
        if len(cards) != 5:
             raise ValueError("OFC 5-card evaluator requires exactly 5 cards.")
        # Проверяем, что все элементы - целые числа
        if not all(isinstance(c, int) for c in cards):
             raise TypeError(f"OFC 5-card evaluator requires integer card representations. Got: {[type(c) for c in cards]}")

        return self._five(cards)

    def _five(self, cards: List[int]) -> int: # Принимает список int
        """
        Performs an evalution given cards in integer form, mapping them to
        a rank in the range [1, 7462], with lower ranks being more powerful.
        """
        # Check for flush using the suit bits (0xF000 mask)
        is_flush = (cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000)

        if is_flush:
            # Calculate the bitmask for ranks involved in the flush
            # Shift right by 16 to isolate the rank bits
            handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
            # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
            prime = CardUtils.prime_product_from_rankbits(handOR)
            # Lookup in the flush table
            rank = self.table.flush_lookup.get(prime)
            if rank is None:
                 # Fallback for safety, though ideally shouldn't happen
                 # print(f"Warning: Flush prime product {prime} not found in flush_lookup. Falling back.")
                 prime_unsuited = CardUtils.prime_product_from_hand(cards)
                 rank = self.table.unsuited_lookup.get(prime_unsuited, LookupTable.MAX_HIGH_CARD + 1)
            return rank
        else:
            # Calculate the prime product for non-flush hands
            # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
            prime = CardUtils.prime_product_from_hand(cards)
            # Lookup in the unsuited table
            return self.table.unsuited_lookup.get(prime, LookupTable.MAX_HIGH_CARD + 1) # Return worst rank if not found

    def get_rank_class(self, hr: int) -> int:
        """
        Returns the class of hand (integer) given the hand rank.
        Uses constants from LookupTable.
        """
        if hr >= 1 and hr <= LookupTable.MAX_STRAIGHT_FLUSH: # 10
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT_FLUSH]
        elif hr <= LookupTable.MAX_FOUR_OF_A_KIND: # 166
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FOUR_OF_A_KIND]
        elif hr <= LookupTable.MAX_FULL_HOUSE: # 322
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FULL_HOUSE]
        elif hr <= LookupTable.MAX_FLUSH: # 1599
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FLUSH]
        elif hr <= LookupTable.MAX_STRAIGHT: # 1609
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT]
        elif hr <= LookupTable.MAX_THREE_OF_A_KIND: # 2467
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_THREE_OF_A_KIND]
        elif hr <= LookupTable.MAX_TWO_PAIR: # 3325
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_TWO_PAIR]
        elif hr <= LookupTable.MAX_PAIR: # 6185
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_PAIR]
        elif hr <= LookupTable.MAX_HIGH_CARD: # 7462
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_HIGH_CARD]
        else:
            # Handle cases where rank might be outside expected range (e.g., fallback value)
            print(f"Warning: Invalid hand rank {hr} in get_rank_class. Returning High Card class.")
            # Возвращаем класс для High Card (обычно 9)
            return LookupTable.MAX_TO_RANK_CLASS.get(LookupTable.MAX_HIGH_CARD, 9)


    def class_to_string(self, class_int: int) -> str:
        """
        Converts the integer class hand score into a human-readable string.
        Uses constants from LookupTable.
        """
        return LookupTable.RANK_CLASS_TO_STRING.get(class_int, "Unknown")