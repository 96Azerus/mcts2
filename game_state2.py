# game_state.py
"""
Определяет класс GameState, управляющий полным состоянием игры
OFC Pineapple для двух игроков.
"""
import copy
import random
import sys
import traceback
from itertools import combinations, permutations
from typing import List, Tuple, Optional, Set, Dict, Any

# Импортируем зависимости из других наших модулей
from card import Card, card_to_str, card_from_str # Используем корневой Card (int)
from deck import Deck # Использует корневой Card (int)
from board import PlayerBoard # Использует обновленный scoring и корневой Card (int)
from scoring import calculate_headsup_score # Использует обновленный scoring

class GameState:
    NUM_PLAYERS = 2

    # Конструктор принимает int карты
    def __init__(self,
                 boards: Optional[List[PlayerBoard]] = None,
                 deck: Optional[Deck] = None,
                 private_discard: Optional[List[List[int]]] = None, # Карты - int
                 dealer_idx: int = 0,
                 current_player_idx: Optional[int] = None,
                 street: int = 1,
                 current_hands: Optional[Dict[int, Optional[List[int]]]] = None, # Карты - int
                 fantasyland_status: Optional[List[bool]] = None,
                 next_fantasyland_status: Optional[List[bool]] = None,
                 fantasyland_cards_to_deal: Optional[List[int]] = None,
                 is_fantasyland_round: bool = False,
                 fantasyland_hands: Optional[List[Optional[List[int]]]] = None, # Карты - int
                 _player_acted_this_street: Optional[List[bool]] = None,
                 _player_finished_round: Optional[List[bool]] = None):

        self.boards: List[PlayerBoard] = boards if boards is not None else [PlayerBoard() for _ in range(self.NUM_PLAYERS)]
        self.deck: Deck = deck if deck is not None else Deck()
        self.private_discard: List[List[int]] = private_discard if private_discard is not None else [[] for _ in range(self.NUM_PLAYERS)]
        self.dealer_idx: int = dealer_idx
        # Определяем первого игрока (слева от дилера)
        self.current_player_idx: int = (1 - dealer_idx) if current_player_idx is None else current_player_idx
        self.street: int = street
        self.current_hands: Dict[int, Optional[List[int]]] = current_hands if current_hands is not None else {i: None for i in range(self.NUM_PLAYERS)}
        self.fantasyland_status: List[bool] = fantasyland_status if fantasyland_status is not None else [False] * self.NUM_PLAYERS
        self.next_fantasyland_status: List[bool] = next_fantasyland_status if next_fantasyland_status is not None else [False] * self.NUM_PLAYERS
        self.fantasyland_cards_to_deal: List[int] = fantasyland_cards_to_deal if fantasyland_cards_to_deal is not None else [0] * self.NUM_PLAYERS
        self.is_fantasyland_round: bool = is_fantasyland_round
        self.fantasyland_hands: List[Optional[List[int]]] = fantasyland_hands if fantasyland_hands is not None else [None] * self.NUM_PLAYERS
        # Отслеживает, сходил ли игрок на ТЕКУЩЕЙ улице
        self._player_acted_this_street: List[bool] = _player_acted_this_street if _player_acted_this_street is not None else [False] * self.NUM_PLAYERS
        # Отслеживает, завершил ли игрок ПОЛНОСТЬЮ свою доску (13 карт)
        self._player_finished_round: List[bool] = _player_finished_round if _player_finished_round is not None else [False] * self.NUM_PLAYERS

    def get_player_board(self, player_idx: int) -> PlayerBoard:
        return self.boards[player_idx]

    def get_player_hand(self, player_idx: int) -> Optional[List[int]]: # Возвращает список int
         """Возвращает текущую руку игрока (обычную или ФЛ)."""
         if self.is_fantasyland_round and self.fantasyland_status[player_idx]:
              return self.fantasyland_hands[player_idx]
         else:
              return self.current_hands.get(player_idx)

    def start_new_round(self, dealer_button_idx: int):
        """Начинает новый раунд, сохраняя статус ФЛ."""
        # Сохраняем статус ФЛ и количество карт из предыдущего раунда
        current_fl_status = list(self.next_fantasyland_status) # Используем next_fantasyland_status
        current_fl_cards = list(self.fantasyland_cards_to_deal) # Используем текущее значение карт

        # Сбрасываем состояние, передавая сохраненный статус ФЛ
        self.__init__(dealer_idx=dealer_button_idx,
                      fantasyland_status=current_fl_status,
                      fantasyland_cards_to_deal=current_fl_cards)
        self.is_fantasyland_round = any(self.fantasyland_status)

        # print(f"DEBUG start_new_round: is_fantasyland_round = {self.is_fantasyland_round}")
        # sys.stdout.flush(); sys.stderr.flush()

        if self.is_fantasyland_round:
            self._deal_fantasyland_hands()
            # Раздаем карты 1й улицы не-ФЛ игрокам сразу
            for i in range(self.NUM_PLAYERS):
                if not self.fantasyland_status[i]:
                    # print(f"DEBUG start_new_round: Calling _deal_street_to_player for non-FL player {i}")
                    # sys.stdout.flush(); sys.stderr.flush()
                    self._deal_street_to_player(i) # Раздаем 5 карт
        else:
            # Обычный раунд, раздаем первому игроку (слева от дилера)
            first_player = 1 - self.dealer_idx
            self.current_player_idx = first_player # Устанавливаем, кто ходит первым
            # print(f"DEBUG start_new_round: Calling _deal_street_to_player for first player {first_player}")
            # sys.stdout.flush(); sys.stderr.flush()
            self._deal_street_to_player(first_player)

    def _deal_street_to_player(self, player_idx: int):
        """Раздает карты для текущей улицы указанному игроку."""
        # print(f"DEBUG: ENTERING _deal_street_to_player for player {player_idx}, street {self.street}")
        # sys.stdout.flush(); sys.stderr.flush()

        # Не раздаем, если у игрока уже есть карты на этой улице или он закончил
        if self._player_finished_round[player_idx] or self.current_hands.get(player_idx) is not None:
             # print(f"DEBUG _deal_street_to_player: Skipping deal for player {player_idx} (finished or already has hand)")
             # sys.stdout.flush(); sys.stderr.flush()
             return

        num_cards = 5 if self.street == 1 else 3
        # print(f"DEBUG _deal_street_to_player: Attempting to deal {num_cards} cards from deck (size {len(self.deck)})")
        # sys.stdout.flush(); sys.stderr.flush()
        try:
            dealt_cards = self.deck.deal(num_cards) # Возвращает список int
            # print(f"DEBUG: Dealt street cards for player {player_idx}, street {self.street}: {[card_to_str(c) for c in dealt_cards]}")
            # sys.stdout.flush(); sys.stderr.flush()
            self.current_hands[player_idx] = dealt_cards
            self._player_acted_this_street[player_idx] = False # Сбрасываем флаг действия на улице
            # print(f"DEBUG _deal_street_to_player: Successfully dealt to player {player_idx}")
            # sys.stdout.flush(); sys.stderr.flush()
        except ValueError as e: # Ловим конкретную ошибку нехватки карт
            print(f"Error dealing street {self.street} to player {player_idx}: {e}")
            sys.stdout.flush(); sys.stderr.flush()
            self.current_hands[player_idx] = [] # Пустая рука при ошибке
        except Exception as e_other: # Ловим другие возможные ошибки
             print(f"Unexpected Error in _deal_street_to_player for player {player_idx}: {e_other}")
             traceback.print_exc()
             sys.stdout.flush(); sys.stderr.flush()
             self.current_hands[player_idx] = []


    def _deal_fantasyland_hands(self):
        """Раздает N карт игрокам в статусе Фантазии."""
        for i in range(self.NUM_PLAYERS):
            if self.fantasyland_status[i]:
                num_cards = self.fantasyland_cards_to_deal[i]
                if num_cards == 0: num_cards = 14 # Стандарт по умолчанию
                try:
                    dealt_cards = self.deck.deal(num_cards) # Возвращает список int
                    # print(f"DEBUG: Dealt fantasyland hand for player {i} ({num_cards} cards): {[card_to_str(c) for c in dealt_cards]}")
                    # sys.stdout.flush(); sys.stderr.flush()
                    self.fantasyland_hands[i] = dealt_cards
                except ValueError as e: # Ловим конкретную ошибку нехватки карт
                    print(f"Error dealing Fantasyland to player {i}: {e}")
                    sys.stdout.flush(); sys.stderr.flush()
                    self.fantasyland_hands[i] = [] # Пустая рука при ошибке

    def get_legal_actions_for_player(self, player_idx: int) -> List[Any]:
        """Возвращает легальные действия для указанного игрока."""
        if self._player_finished_round[player_idx]: return []

        # Фантазия
        if self.is_fantasyland_round and self.fantasyland_status[player_idx]:
            hand = self.fantasyland_hands[player_idx]
            # Для MCTS/AI, ФЛ действие - это сама рука (солвер разберется)
            # Возвращаем в формате, который может обработать MCTSNode/MCTSAgent
            # Например, кортеж с маркером и рукой
            return [("FANTASYLAND_INPUT", hand)] if hand else []

        # Обычный ход
        hand = self.current_hands.get(player_idx)
        if not hand: return [] # Нет карт для хода

        if self.street == 1:
            return self._get_legal_actions_street1(player_idx, hand) if len(hand) == 5 else []
        else: # Улицы 2-5
            return self._get_legal_actions_pineapple(player_idx, hand) if len(hand) == 3 else []

    def _get_legal_actions_street1(self, player_idx: int, hand: List[int]) -> List[Tuple[List[Tuple[int, str, int]], List[int]]]: # Карты - int
        """Генерирует ВСЕ легальные действия для первой улицы (размещение 5 карт)."""
        board = self.boards[player_idx]
        available_slots = board.get_available_slots()
        if len(available_slots) < 5: return []

        actions = []
        hand_list = list(hand) # Копия руки (список int)

        # Ограничиваем комбинации для производительности
        MAX_SLOT_COMBOS = 500 # Можно настроить
        MAX_CARD_PERMS = 120 # 5!

        slot_combinations = list(combinations(available_slots, 5))
        if len(slot_combinations) > MAX_SLOT_COMBOS:
             slot_combinations = random.sample(slot_combinations, MAX_SLOT_COMBOS)

        # Генерируем перестановки карт один раз
        card_permutations = list(permutations(hand_list))

        # Перебираем комбинации слотов и (опционально) перестановок карт
        for slot_combination in slot_combinations:
            # Используем одну случайную перестановку карт для каждой комбинации слотов (эвристика)
            card_permutation = random.choice(card_permutations)
            # Или можно перебирать все: for card_permutation in card_permutations:

            placement = []
            valid_placement = True
            temp_placed_slots = set() # Проверка уникальности слотов внутри действия
            for i in range(5):
                card_int = card_permutation[i] # Карта - int
                slot_info = slot_combination[i]
                row_name, index = slot_info
                # Проверка, что слот не используется дважды в ОДНОМ действии
                if slot_info in temp_placed_slots:
                    valid_placement = False; break
                temp_placed_slots.add(slot_info)
                placement.append((card_int, row_name, index)) # Добавляем int карту

            if valid_placement:
                actions.append((placement, [])) # Пустой сброс для улицы 1

        return actions

    def _get_legal_actions_pineapple(self, player_idx: int, hand: List[int]) -> List[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]]: # Карты - int
        """Генерирует действия для улиц 2-5."""
        board = self.boards[player_idx]
        available_slots = board.get_available_slots()
        if len(available_slots) < 2: return []

        actions = []
        hand_list = list(hand) # Копия руки (список int)

        # Перебираем карту для сброса
        for i in range(3):
            discarded_card_int = hand_list[i] # Карта - int
            cards_to_place = [hand_list[j] for j in range(3) if i != j]
            card1_int, card2_int = cards_to_place[0], cards_to_place[1] # Карты - int

            # Перебираем комбинации из 2 доступных слотов
            for slot1_info, slot2_info in combinations(available_slots, 2):
                row1, idx1 = slot1_info
                row2, idx2 = slot2_info
                # Два варианта размещения карт в выбранные слоты
                actions.append(((card1_int, row1, idx1), (card2_int, row2, idx2), discarded_card_int))
                # Если слоты разные, добавляем обратный вариант
                if slot1_info != slot2_info:
                     actions.append(((card2_int, row1, idx1), (card1_int, row2, idx2), discarded_card_int))
        return actions

    def apply_action(self, player_idx: int, action: Any):
        """
        Применяет легальное действие для УКАЗАННОГО игрока.
        Возвращает НОВОЕ состояние игры.
        ВАЖНО: Эта функция НЕ управляет очередностью ходов или завершением раунда.
        Ожидает карты в формате int.
        """
        new_state = self.copy()
        board = new_state.boards[player_idx]

        # Проверка, что это не ход ФЛ
        if new_state.is_fantasyland_round and new_state.fantasyland_status[player_idx]:
             print(f"Warning: apply_action called for Fantasyland player {player_idx}. Use apply_fantasyland_placement.")
             # Можно вернуть ошибку или текущее состояние
             return self # Возвращаем старое состояние

        current_hand = new_state.current_hands.get(player_idx)
        if not current_hand:
            print(f"Error: apply_action called for player {player_idx} but no hand found.")
            return self # Возвращаем старое состояние

        if new_state.street == 1:
            # --- Улица 1 ---
            if len(current_hand) != 5:
                print(f"Error: apply_action street 1 wrong hand size {len(current_hand)}")
                return self
            try:
                placements, _ = action # Действие = ([(card_int, row, idx), ...], [])
                if len(placements) != 5:
                    print(f"Error: apply_action street 1 wrong placement size {len(placements)}")
                    return self
            except (TypeError, ValueError):
                 print(f"Error: Invalid action format for street 1: {action}")
                 return self

            success = True
            placed_cards_in_action = set()
            current_hand_set = set(current_hand)
            for card_int, row, index in placements:
                # Проверяем, что карта из руки и не дублируется в действии
                if card_int not in current_hand_set or card_int in placed_cards_in_action:
                    print(f"Error: Invalid card {card_to_str(card_int)} in street 1 placement.")
                    success = False; break
                # Пытаемся добавить карту на доску
                if not board.add_card(card_int, row, index):
                    print(f"Error: Failed to add card {card_to_str(card_int)} to {row}[{index}] in street 1.")
                    success = False; break
                placed_cards_in_action.add(card_int)

            if not success:
                # Откатываем изменения на доске, если что-то пошло не так
                # Проще вернуть исходное состояние, так как откат сложен
                print(f"Error applying street 1 action for player {player_idx}. Reverting.")
                return self # Возвращаем старое состояние

            # Успешное размещение
            new_state.current_hands[player_idx] = None # Убираем руку
            new_state._player_acted_this_street[player_idx] = True # Отмечаем действие
            # Проверяем завершение доски и обновляем статус ФЛ
            if board.is_complete():
                new_state._player_finished_round[player_idx] = True
                new_state._check_foul_and_update_fl_status(player_idx)

        else:
            # --- Улицы 2-5 (Pineapple) ---
            if len(current_hand) != 3:
                print(f"Error: apply_action pineapple wrong hand size {len(current_hand)}")
                return self
            try:
                place1, place2, discarded_card_int = action # ((card1_int, r1, i1), (card2_int, r2, i2), discard_int)
                card1_int, row1, idx1 = place1
                card2_int, row2, idx2 = place2
            except (TypeError, ValueError):
                 print(f"Error: Invalid action format for pineapple: {action}")
                 return self

            # Проверяем карты действия
            action_cards = {card1_int, card2_int, discarded_card_int}
            current_hand_set = set(current_hand)
            if len(action_cards) != 3 or action_cards != current_hand_set:
                print(f"Error: Action cards mismatch hand for player {player_idx}.")
                print(f"Action cards: {[card_to_str(c) for c in action_cards]}")
                print(f"Hand cards: {[card_to_str(c) for c in current_hand_set]}")
                return self

            # Пытаемся разместить карты
            success1 = board.add_card(card1_int, row1, idx1)
            success2 = board.add_card(card2_int, row2, idx2)

            if not success1 or not success2:
                print(f"Error applying pineapple action for player {player_idx}: failed to add cards.")
                # Откатываем первое добавление, если оно было успешным, а второе нет
                if success1 and not success2:
                    board.remove_card(row1, idx1) # Откатываем только если первая карта была добавлена
                return self # Возвращаем старое состояние при любой ошибке добавления

            # Успешное размещение
            new_state.private_discard[player_idx].append(discarded_card_int) # Добавляем в сброс
            new_state.current_hands[player_idx] = None # Убираем руку
            new_state._player_acted_this_street[player_idx] = True # Отмечаем действие
            # Проверяем завершение доски и обновляем статус ФЛ
            if board.is_complete():
                new_state._player_finished_round[player_idx] = True
                new_state._check_foul_and_update_fl_status(player_idx)

        return new_state

    def apply_fantasyland_placement(self, player_idx: int, placement: Dict[str, List[int]], discarded: List[int]): # Карты - int
        """Применяет результат FantasylandSolver к доске игрока."""
        new_state = self.copy()
        board = new_state.boards[player_idx]

        # Проверки
        if not new_state.is_fantasyland_round or not new_state.fantasyland_status[player_idx] or not new_state.fantasyland_hands[player_idx]:
            print(f"Error: apply_fantasyland_placement called incorrectly for player {player_idx}.")
            return self # Возвращаем старое состояние

        original_hand = set(new_state.fantasyland_hands[player_idx])
        placed_cards_in_placement = set(c for row in placement.values() for c in row)
        discarded_set = set(discarded)
        expected_discard_count = len(original_hand) - 13

        # Проверка консистентности данных
        if len(placed_cards_in_placement) != 13 or \
           len(discarded) != expected_discard_count or \
           not placed_cards_in_placement.union(discarded_set) == original_hand or \
           not placed_cards_in_placement.isdisjoint(discarded_set):
            print(f"Error: Invalid Fantasyland placement/discard data for player {player_idx}.")
            print(f"Hand:{len(original_hand)}, Placed:{len(placed_cards_in_placement)}, Discard:{len(discarded)}, ExpectedDiscard:{expected_discard_count}")
            # Фол в случае неконсистентности
            return new_state.apply_fantasyland_foul(player_idx, new_state.fantasyland_hands[player_idx])

        # Устанавливаем доску
        try:
            board.set_full_board(placement['top'], placement['middle'], placement['bottom'])
        except ValueError as e:
            print(f"Error setting FL board for player {player_idx}: {e}")
            # Фол в случае ошибки установки
            return new_state.apply_fantasyland_foul(player_idx, new_state.fantasyland_hands[player_idx])

        # Успешное размещение
        new_state.private_discard[player_idx].extend(discarded)
        new_state.fantasyland_hands[player_idx] = None # Убираем руку ФЛ
        new_state._player_finished_round[player_idx] = True # Игрок завершил раунд
        # Проверяем фол и статус ФЛ ПОСЛЕ установки доски
        new_state._check_foul_and_update_fl_status(player_idx)
        return new_state

    def apply_fantasyland_foul(self, player_idx: int, hand_to_discard: List[int]): # Карты - int
        """Применяет фол в Fantasyland."""
        new_state = self.copy()
        board = new_state.boards[player_idx]
        board.is_foul = True # Устанавливаем флаг фола
        # Опционально: Очищаем доску от карт при фоле
        # board.rows = {name: [None] * capacity for name, capacity in PlayerBoard.ROW_CAPACITY.items()}
        # board._cards_placed = 0
        # board._is_complete = False # Технически не полная, но раунд завершен
        # board._reset_caches() # Сбрасываем кэши

        new_state.private_discard[player_idx].extend(hand_to_discard) # Сбрасываем всю руку
        new_state.fantasyland_hands[player_idx] = None # Убираем руку ФЛ
        new_state._player_finished_round[player_idx] = True # Игрок завершил раунд (фолом)
        new_state.next_fantasyland_status[player_idx] = False # Фол не дает ФЛ
        new_state.fantasyland_cards_to_deal[player_idx] = 0
        return new_state

    def _check_foul_and_update_fl_status(self, player_idx: int):
        """
        Проверяет фол и обновляет статус FL для игрока, завершившего доску.
        Использует методы PlayerBoard, которые вызывают обновленный scoring.
        """
        board = self.boards[player_idx]
        if not board.is_complete(): return # Ничего не делаем, если доска не полная

        is_foul = board.check_and_set_foul() # Проверяем фол и устанавливаем флаг

        self.next_fantasyland_status[player_idx] = False # Сбрасываем по умолчанию
        self.fantasyland_cards_to_deal[player_idx] = 0

        if not is_foul:
            if self.fantasyland_status[player_idx]: # Если игрок БЫЛ в ФЛ
                if board.check_fantasyland_stay_conditions(): # Проверяем удержание
                    self.next_fantasyland_status[player_idx] = True
                    self.fantasyland_cards_to_deal[player_idx] = 14 # Повтор всегда 14 карт
            else: # Если игрок НЕ был в ФЛ
                fl_cards = board.get_fantasyland_qualification_cards() # Проверяем вход
                if fl_cards > 0:
                    self.next_fantasyland_status[player_idx] = True
                    self.fantasyland_cards_to_deal[player_idx] = fl_cards

    def is_round_over(self) -> bool:
        """Проверяет, завершили ли все игроки свою часть раунда."""
        return all(self._player_finished_round)

    def get_terminal_score(self) -> int:
        """
        Возвращает счет раунда с точки зрения Игрока 0.
        Использует calculate_headsup_score из scoring.py.
        """
        if not self.is_round_over(): return 0
        # Убедимся, что фолы проверены для обеих досок перед подсчетом
        # check_and_set_foul вызывается внутри calculate_headsup_score через методы board
        # for board in self.boards:
        #      if board.is_complete(): board.check_and_set_foul() # Это уже делается в calculate_headsup_score

        # Вызываем обновленную функцию подсчета
        return calculate_headsup_score(self.boards[0], self.boards[1])

    def get_known_dead_cards(self, perspective_player_idx: int) -> Set[int]: # Возвращает set int
         """Возвращает набор карт (int), известных игроку как вышедшие из игры."""
         dead_cards = set()
         # Карты на всех досках
         for board in self.boards:
             for row_name in board.ROW_NAMES:
                 for card_int in board.rows[row_name]:
                     if card_int is not None: dead_cards.add(card_int)
         # Карты в руке игрока (если есть)
         player_hand = self.get_player_hand(perspective_player_idx)
         if player_hand: dead_cards.update(player_hand)
         # Карты в личном сбросе игрока
         dead_cards.update(self.private_discard[perspective_player_idx])
         return dead_cards

    def get_state_representation(self) -> tuple:
        """Возвращает неизменяемое представление состояния для MCTS."""
        # Используем board.get_board_state_tuple(), который возвращает строки
        board_tuples = tuple(b.get_board_state_tuple() for b in self.boards)

        # Представление рук: есть ли карты у игрока (True/False)
        current_hands_tuple = tuple(bool(self.current_hands.get(i)) for i in range(self.NUM_PLAYERS))
        fantasyland_hands_tuple = tuple(bool(self.fantasyland_hands[i]) for i in range(self.NUM_PLAYERS))

        # Представление сброса: кортеж кортежей отсортированных карт (строки)
        discard_tuple = tuple(tuple(sorted(card_to_str(c) for c in p_discard)) for p_discard in self.private_discard)

        return (
            board_tuples,
            discard_tuple, # Добавляем сброс для большей уникальности
            self.dealer_idx,
            self.current_player_idx,
            self.street,
            tuple(self.fantasyland_status),
            self.is_fantasyland_round,
            current_hands_tuple,
            fantasyland_hands_tuple,
            tuple(self._player_acted_this_street),
            tuple(self._player_finished_round)
        )

    def copy(self) -> 'GameState':
        """Создает глубокую копию состояния."""
        # copy.deepcopy должен справиться с копированием PlayerBoard и Deck
        return copy.deepcopy(self)

    def __hash__(self):
        # Хэшируем неизменяемое представление
        return hash(self.get_state_representation())

    def __eq__(self, other):
        if not isinstance(other, GameState): return NotImplemented
        # Сравниваем неизменяемые представления
        return self.get_state_representation() == other.get_state_representation()

    # --- Методы to_dict и from_dict используют card_to_str/card_from_str ---
    # --- ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ ---
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует состояние в словарь для JSON-сериализации."""
        boards_dict = []
        for board in self.boards:
            board_data = {}
            for row_name in PlayerBoard.ROW_NAMES:
                # Используем card_to_str для преобразования int Card в строку
                board_data[row_name] = [card_to_str(c) if c is not None else "__" for c in board.rows[row_name]]
            board_data['_cards_placed'] = board._cards_placed
            board_data['is_foul'] = board.is_foul
            board_data['_is_complete'] = board._is_complete
            boards_dict.append(board_data)

        # print(f"DEBUG to_dict: current_hands before str conversion: { {idx: [card_to_str(c) for c in hand] if hand else None for idx, hand in self.current_hands.items()} }")
        # print(f"DEBUG to_dict: fantasyland_hands before str conversion: { [[card_to_str(c) for c in hand] if hand else None for hand in self.fantasyland_hands] }")
        # sys.stdout.flush(); sys.stderr.flush()

        return {
            "boards": boards_dict,
            "private_discard": [[card_to_str(c) for c in p_discard] for p_discard in self.private_discard],
            "dealer_idx": self.dealer_idx,
            "current_player_idx": self.current_player_idx,
            "street": self.street,
            "current_hands": {idx: [card_to_str(c) for c in hand] if hand else None for idx, hand in self.current_hands.items()},
            "fantasyland_status": self.fantasyland_status,
            "next_fantasyland_status": self.next_fantasyland_status,
            "fantasyland_cards_to_deal": self.fantasyland_cards_to_deal,
            "is_fantasyland_round": self.is_fantasyland_round,
            "fantasyland_hands": [[card_to_str(c) for c in hand] if hand else None for hand in self.fantasyland_hands],
            "_player_acted_this_street": self._player_acted_this_street,
            "_player_finished_round": self._player_finished_round,
            # Не сохраняем deck, он восстанавливается из известных карт
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        """Восстанавливает состояние из словаря."""
        boards = []
        all_known_cards_ints = set() # Собираем int представления известных карт
        num_players = len(data.get("boards", []))
        if num_players == 0: num_players = cls.NUM_PLAYERS # По умолчанию 2

        for board_data in data.get("boards", [{} for _ in range(num_players)]):
            board = PlayerBoard()
            cards_on_board = 0
            for row_name in PlayerBoard.ROW_NAMES:
                cards = []
                for card_str in board_data.get(row_name, []):
                    if card_str != "__":
                        try:
                            # Используем card_from_str для получения int Card
                            card_int = card_from_str(card_str)
                            cards.append(card_int)
                            all_known_cards_ints.add(card_int)
                            cards_on_board += 1
                        except ValueError:
                             print(f"Warning: Invalid card string '{card_str}' in saved board state.")
                             cards.append(None)
                    else: cards.append(None)
                capacity = PlayerBoard.ROW_CAPACITY[row_name]
                # Дополняем None до нужной емкости
                cards.extend([None] * (capacity - len(cards)))
                board.rows[row_name] = cards[:capacity] # Обрезаем на всякий случай
            # Восстанавливаем внутренние счетчики и флаги
            board._cards_placed = board_data.get('_cards_placed', cards_on_board)
            board.is_foul = board_data.get('is_foul', False)
            board._is_complete = board_data.get('_is_complete', board._cards_placed == 13)
            # Сбрасываем кэши при загрузке, они пересчитаются
            board._reset_caches()
            boards.append(board)

        private_discard = []
        for p_discard_strs in data.get("private_discard", [[] for _ in range(num_players)]):
            p_discard = []
            for cs in p_discard_strs:
                 try:
                      card_int = card_from_str(cs)
                      p_discard.append(card_int)
                      all_known_cards_ints.add(card_int)
                 except ValueError: print(f"Warning: Invalid card string '{cs}' in saved private discard.")
            private_discard.append(p_discard)

        current_hands = {}
        for idx_str, hand_strs in data.get("current_hands", {}).items():
             idx = int(idx_str)
             if hand_strs:
                  hand = []
                  for cs in hand_strs:
                       if cs == 'InvalidCard': continue # Игнорируем
                       try:
                            card_int = card_from_str(cs)
                            hand.append(card_int)
                            all_known_cards_ints.add(card_int)
                       except ValueError: print(f"Warning: Invalid card string '{cs}' in saved current hand.")
                  current_hands[idx] = hand
             else: current_hands[idx] = None
        # Убедимся, что ключи есть для всех игроков
        for i in range(num_players):
             if i not in current_hands: current_hands[i] = None

        fantasyland_hands = []
        for hand_strs in data.get("fantasyland_hands", [None]*num_players):
            if hand_strs:
                hand = []
                for cs in hand_strs:
                     if cs == 'InvalidCard': continue # Игнорируем
                     try:
                          card_int = card_from_str(cs)
                          hand.append(card_int)
                          all_known_cards_ints.add(card_int)
                     except ValueError: print(f"Warning: Invalid card string '{cs}' in saved fantasyland hand.")
                fantasyland_hands.append(hand)
            else: fantasyland_hands.append(None)

        # Восстанавливаем колоду
        # FULL_DECK_CARDS теперь содержит int представления
        remaining_cards_ints = Deck.FULL_DECK_CARDS - all_known_cards_ints
        deck = Deck(cards=remaining_cards_ints)

        # Значения по умолчанию для списков bool/int
        default_bool_list = [False] * num_players
        default_int_list = [0] * num_players

        # Получаем остальные данные из словаря или используем значения по умолчанию
        dealer_idx = data.get("dealer_idx", 0)
        current_player_idx = data.get("current_player_idx", 1 - dealer_idx) # Вычисляем, если не задан
        street = data.get("street", 1)
        fantasyland_status = data.get("fantasyland_status", list(default_bool_list))
        next_fantasyland_status = data.get("next_fantasyland_status", list(default_bool_list))
        fantasyland_cards_to_deal = data.get("fantasyland_cards_to_deal", list(default_int_list))
        is_fantasyland_round = data.get("is_fantasyland_round", any(fantasyland_status)) # Вычисляем, если не задан
        _player_acted_this_street = data.get("_player_acted_this_street", list(default_bool_list))
        _player_finished_round = data.get("_player_finished_round", list(default_bool_list))


        return cls(
            boards=boards, deck=deck, private_discard=private_discard,
            dealer_idx=dealer_idx,
            current_player_idx=current_player_idx,
            street=street,
            current_hands=current_hands,
            fantasyland_status=fantasyland_status,
            next_fantasyland_status=next_fantasyland_status,
            fantasyland_cards_to_deal=fantasyland_cards_to_deal,
            is_fantasyland_round=is_fantasyland_round,
            fantasyland_hands=fantasyland_hands,
            _player_acted_this_street=_player_acted_this_street,
            _player_finished_round=_player_finished_round
        )