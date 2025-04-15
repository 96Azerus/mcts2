# deck.py
"""
Реализация колоды карт с использованием set для эффективности.
"""
import random
import sys
import traceback
from typing import List, Set, Optional

# Импортируем корневой Card и его утилиты
from card import Card, card_from_str, card_to_str # Используем корневой card.py

class Deck:
    """Представляет колоду карт для OFC."""
    # Создаем полный набор строк карт один раз
    FULL_DECK_STRS = {r + s for r in Card.STR_RANKS for s in Card.SUIT_CHAR_TO_INT.keys()}
    # Создаем полный набор целочисленных представлений карт один раз
    FULL_DECK_CARDS: Set[int] = set()
    print("DEBUG Deck: Initializing FULL_DECK_CARDS...")
    sys.stdout.flush(); sys.stderr.flush()
    initialization_errors = 0
    for cs in FULL_DECK_STRS:
        try:
            # Используем card_from_str (алиас Card.new) для получения int
            card_int = card_from_str(cs)
            # Проверка на всякий случай (хотя Card.new должен быть надежным)
            if not isinstance(card_int, int) or card_int < 0:
                 print(f"ERROR Deck Init: Card('{cs}') created invalid int representation: {card_int}!")
                 initialization_errors += 1
            else:
                 FULL_DECK_CARDS.add(card_int)
        except Exception as e:
            print(f"ERROR Deck Init: Failed to create Card('{cs}'): {e}")
            traceback.print_exc()
            initialization_errors += 1

    print(f"DEBUG Deck: Initialized FULL_DECK_CARDS with {len(FULL_DECK_CARDS)} cards. Errors: {initialization_errors}")
    if len(FULL_DECK_CARDS) != 52:
        print(f"CRITICAL ERROR: FULL_DECK_CARDS contains {len(FULL_DECK_CARDS)} cards instead of 52!")
    sys.stdout.flush(); sys.stderr.flush()


    def __init__(self, cards: Optional[Set[int]] = None): # Тип карт теперь int
        """
        Инициализирует колоду.
        Если cards is None, создает полную колоду.
        Иначе использует переданный набор карт (копируя его).
        """
        if cards is None:
            # Копируем из предсозданного набора int
            self.cards: Set[int] = self.FULL_DECK_CARDS.copy()
        else:
            # Важно копировать переданный set
            self.cards: Set[int] = cards.copy()

    def deal(self, n: int) -> List[int]: # Возвращает список int
        """Раздает n случайных карт из колоды и удаляет их."""
        current_len = len(self.cards)
        n_req = n

        if n <= 0: return []
        if n > current_len:
            print(f"Warning: Trying to deal {n_req} cards, only {current_len} left. Dealing {current_len}.")
            n = current_len
        if n == 0: return []

        try:
            # Преобразуем set в list для random.sample
            card_list = list(self.cards)
            if n > len(card_list): # Дополнительная проверка
                 print(f"Error: Requested {n} cards, but only {len(card_list)} available in list form.")
                 n = len(card_list)
                 if n == 0: return []
            # Выбираем n случайных карт (int)
            dealt_cards = random.sample(card_list, n)
            # Удаляем розданные карты из set (эффективно)
            self.cards.difference_update(dealt_cards)
            return dealt_cards
        except Exception as e:
             print(f"ERROR in Deck.deal: {e}")
             traceback.print_exc()
             sys.stdout.flush(); sys.stderr.flush()
             # Возвращаем пустой список в случае ошибки
             return []

    def remove(self, cards_to_remove: List[int]): # Принимает список int
        """Удаляет конкретные карты из колоды."""
        self.cards.difference_update(cards_to_remove)

    def add(self, cards_to_add: List[int]): # Принимает список int
        """Добавляет карты обратно в колоду (например, при откате хода)."""
        self.cards.update(cards_to_add)

    def get_remaining_cards(self) -> List[int]: # Возвращает список int
        """Возвращает список оставшихся карт."""
        return list(self.cards)

    def copy(self) -> 'Deck':
        """Создает копию колоды."""
        # Передаем копию текущего набора карт (int)
        return Deck(self.cards.copy())

    def __len__(self) -> int:
        """Возвращает количество карт в колоде."""
        return len(self.cards)

    def __contains__(self, card: int) -> bool: # Проверяет наличие int
        """Проверяет наличие карты в колоде O(1)."""
        return card in self.cards

    def __str__(self) -> str:
        """Строковое представление колоды (для отладки)."""
        return f"Deck({len(self.cards)} cards)"

    def __repr__(self) -> str:
        return self.__str__()