# board.py
"""
Представление доски одного игрока.
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import copy

# Импортируем корневой Card и его утилиты
from card import Card, card_to_str, RANK_MAP, STR_RANKS # Добавили RANK_MAP, STR_RANKS

# Импортируем НУЖНЫЕ функции из обновленного scoring
from scoring import (get_hand_rank_safe, check_board_foul,
                     get_fantasyland_entry_cards, check_fantasyland_stay,
                     get_row_royalty, RANK_CLASS_HIGH_CARD) # RANK_CLASS_HIGH_CARD все еще нужен для _get_rank

class PlayerBoard:
    ROW_CAPACITY: Dict[str, int] = {'top': 3, 'middle': 5, 'bottom': 5}
    ROW_NAMES: List[str] = ['top', 'middle', 'bottom']

    def __init__(self):
        # Инициализируем ряды пустыми списками None
        self.rows: Dict[str, List[Optional[Card]]] = {
            name: [None] * capacity for name, capacity in self.ROW_CAPACITY.items()
        }
        self._cards_placed: int = 0
        self.is_foul: bool = False
        # Кэши для рангов и роялти
        self._cached_ranks: Dict[str, Optional[int]] = {name: None for name in self.ROW_NAMES}
        self._cached_royalties: Dict[str, Optional[int]] = {name: None for name in self.ROW_NAMES}
        self._is_complete: bool = False

    def _get_next_index(self, row_name: str) -> Optional[int]:
        """Находит индекс первого None в ряду."""
        try:
            # Используем list.index() для поиска первого None
            return self.rows[row_name].index(None)
        except ValueError:
            return None # Ряд полон (None не найден)

    def add_card(self, card: Card, row_name: str, index: int) -> bool:
        """
        Добавляет карту в УКАЗАННЫЙ слот.
        Возвращает True при успехе, False при неудаче (слот занят, индекс неверный).
        """
        if row_name not in self.ROW_NAMES:
            # print(f"Error: Invalid row name '{row_name}'")
            return False

        capacity = self.ROW_CAPACITY[row_name]
        if not (0 <= index < capacity):
            # print(f"Error: Index {index} out of bounds for row '{row_name}' (0-{capacity-1}).")
            return False

        if self.rows[row_name][index] is not None:
            # print(f"Warning: Slot {row_name}[{index}] is already occupied. Cannot add {card_to_str(card)}.")
            return False

        # Добавляем карту (карта уже должна быть int)
        self.rows[row_name][index] = card
        self._cards_placed += 1
        self._is_complete = (self._cards_placed == 13)

        # Сбрасываем кэши при изменении доски
        self._reset_caches()
        # Фол будет пересчитан при завершении доски
        self.is_foul = False
        return True

    def remove_card(self, row_name: str, index: int) -> Optional[Card]:
         """Удаляет карту из указанного слота (для UI отмены хода)."""
         if row_name not in self.ROW_NAMES or not (0 <= index < self.ROW_CAPACITY[row_name]):
              return None
         card = self.rows[row_name][index]
         if card is not None:
              self.rows[row_name][index] = None
              self._cards_placed -= 1
              self._is_complete = False
              self._reset_caches()
              self.is_foul = False
         return card


    def set_full_board(self, top: List[Card], middle: List[Card], bottom: List[Card]):
        """Устанавливает доску из готовых списков карт (для Фантазии)."""
        if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
            raise ValueError("Incorrect number of cards for setting full board.")

        # Проверяем уникальность карт перед установкой (карты - int)
        all_cards = top + middle + bottom
        if len(all_cards) != len(set(all_cards)):
             raise ValueError("Duplicate cards provided for setting full board.")

        self.rows['top'] = list(top)
        self.rows['middle'] = list(middle)
        self.rows['bottom'] = list(bottom)

        self._cards_placed = 13
        self._is_complete = True
        # Сбрасываем кэши и проверяем фол
        self._reset_caches()
        self.check_and_set_foul() # Проверяем фол сразу после установки

    def get_row_cards(self, row_name: str) -> List[Card]:
        """Возвращает список карт в ряду (без None)."""
        if row_name not in self.rows: return []
        # Возвращаем только не-None карты (они уже int)
        return [card for card in self.rows[row_name] if card is not None]

    def is_row_full(self, row_name: str) -> bool:
        """Проверяет, заполнен ли ряд."""
        if row_name not in self.rows: return False
        # Проверяем, что все слоты в ряду не None
        return all(slot is not None for slot in self.rows[row_name])

    def get_available_slots(self) -> List[Tuple[str, int]]:
        """Возвращает список доступных слотов ('row_name', index)."""
        slots = []
        for row_name in self.ROW_NAMES:
            for i, card in enumerate(self.rows[row_name]):
                if card is None:
                    slots.append((row_name, i))
        return slots

    def get_total_cards(self) -> int:
        """Возвращает количество размещенных карт."""
        return self._cards_placed

    def is_complete(self) -> bool:
        """Проверяет, размещены ли все 13 карт."""
        return self._is_complete

    def _reset_caches(self):
         """Сбрасывает внутренние кэши рангов и роялти."""
         self._cached_ranks = {name: None for name in self.ROW_NAMES}
         self._cached_royalties = {name: None for name in self.ROW_NAMES}

    def _get_rank(self, row_name: str) -> int:
        """
        Получает ранг руки ряда (из кэша или вычисляет).
        Использует get_hand_rank_safe из scoring.py.
        """
        if row_name not in self.ROW_NAMES: return RANK_CLASS_HIGH_CARD + 300 # Худший ранг

        if self._cached_ranks[row_name] is None:
             # Передаем список с None, get_hand_rank_safe обработает
             cards_with_none = self.rows[row_name]
             # Вызываем обновленную функцию из scoring
             self._cached_ranks[row_name] = get_hand_rank_safe(cards_with_none)

        return self._cached_ranks[row_name]

    def check_and_set_foul(self) -> bool:
        """
        Проверяет фол и устанавливает флаг is_foul. Вызывать только на полной доске.
        Использует check_board_foul из scoring.py.
        """
        if not self.is_complete():
            self.is_foul = False # Не фол, пока не полная
            return False

        # Используем функцию из scoring.py, передавая списки с None
        self.is_foul = check_board_foul(
            self.rows['top'],
            self.rows['middle'],
            self.rows['bottom']
        )
        # Если фол, обнуляем кэш роялти
        if self.is_foul:
             self._cached_royalties = {'top': 0, 'middle': 0, 'bottom': 0}
        return self.is_foul

    def get_royalties(self) -> Dict[str, int]:
        """
        Считает и возвращает роялти для каждой линии (используя кэш).
        Использует get_row_royalty из scoring.py.
        """
        # Сначала проверяем фол, если доска полная
        # check_and_set_foul обновит self.is_foul и кэш роялти при необходимости
        if self.is_complete() and self.check_and_set_foul():
            return {'top': 0, 'middle': 0, 'bottom': 0}

        # Пересчитываем роялти, если нужно
        for row_name in self.ROW_NAMES:
             if self._cached_royalties[row_name] is None:
                 # Передаем список с None, get_row_royalty обработает
                 cards_with_none = self.rows[row_name]
                 # Вызываем обновленную функцию из scoring
                 self._cached_royalties[row_name] = get_row_royalty(cards_with_none, row_name)

        # Возвращаем копию кэша
        # Кэш уже должен быть обнулен, если был фол
        return self._cached_royalties.copy()


    def get_total_royalty(self) -> int:
        """Возвращает сумму роялти по всем линиям."""
        # Вызов get_royalties() обновит кэш и учтет фол, если нужно
        royalties = self.get_royalties()
        return sum(royalties.values())

    def get_fantasyland_qualification_cards(self) -> int:
        """
        Возвращает кол-во карт для ФЛ (0 если нет). Проверяет фол.
        Использует get_fantasyland_entry_cards из scoring.py.
        """
        if not self.is_complete(): return 0
        if self.check_and_set_foul(): return 0 # Если фол, ФЛ невозможен
        # Вызываем обновленную функцию из scoring
        return get_fantasyland_entry_cards(self.rows['top'])

    def check_fantasyland_stay_conditions(self) -> bool:
        """
        Проверяет условия удержания ФЛ. Проверяет фол.
        Использует check_fantasyland_stay из scoring.py.
        """
        if not self.is_complete(): return False
        if self.check_and_set_foul(): return False # Если фол, удержать ФЛ нельзя
        # Вызываем обновленную функцию из scoring
        return check_fantasyland_stay(
            self.rows['top'],
            self.rows['middle'],
            self.rows['bottom']
        )

    def get_board_state_tuple(self) -> Tuple[Tuple[Optional[str], ...], ...]:
        """
        Возвращает неизменяемое представление доски (строки карт, включая '__').
        Сортирует строки внутри рядов для каноничности.
        """
        # Используем RANK_MAP и STR_RANKS из корневого card.py для сортировки
        key_func = lambda s: RANK_MAP.get(s[0], -1) if s != "__" and s[0] in RANK_MAP else float('inf')

        rows_as_str = {}
        for r_name in self.ROW_NAMES:
             row_str_list = []
             for card_int in self.rows[r_name]:
                  # Преобразуем int Card в строку или "__"
                  row_str_list.append(card_to_str(card_int) if card_int is not None else "__")
             # Сортируем строки карт внутри рядов для каноничности, '__' идут в конец
             rows_as_str[r_name] = tuple(sorted(row_str_list, key=key_func))

        return (rows_as_str['top'], rows_as_str['middle'], rows_as_str['bottom'])

    def copy(self) -> 'PlayerBoard':
        """Создает глубокую копию доски."""
        # Используем copy.deepcopy для надежности копирования списков карт (int)
        new_board = PlayerBoard()
        # Копируем ряды глубоко (списки int копируются по значению)
        new_board.rows = {r: list(cards) for r, cards in self.rows.items()}
        new_board._cards_placed = self._cards_placed
        new_board.is_foul = self.is_foul
        # Копируем кэши (они содержат простые типы)
        new_board._cached_ranks = self._cached_ranks.copy()
        new_board._cached_royalties = self._cached_royalties.copy()
        new_board._is_complete = self._is_complete
        return new_board

    def __str__(self) -> str:
        """Строковое представление доски."""
        s = ""
        max_len = max(len(self.rows[r_name]) for r_name in self.ROW_NAMES)
        for r_name in self.ROW_NAMES:
            # Преобразуем int Card в строку или "__"
            row_str = [card_to_str(c) if c is not None else "__" for c in self.rows[r_name]]
            # Дополняем пробелами для выравнивания
            row_str += ["  "] * (max_len - len(row_str))
            s += f"{r_name.upper():<6}: " + " ".join(row_str) + "\n"
        if self.is_complete():
             s += f"Complete: Yes, Foul: {self.is_foul}\n"
             # Вызываем get_royalties для обновления и получения актуальных значений
             royalties_dict = self.get_royalties()
             s += f"Royalties: {sum(royalties_dict.values())} {royalties_dict}\n"
        else:
             s += f"Complete: No, Cards: {self._cards_placed}\n"
        return s.strip()

    def __repr__(self) -> str:
         return f"PlayerBoard(Cards={self._cards_placed}, Complete={self._is_complete}, Foul={self.is_foul})"
