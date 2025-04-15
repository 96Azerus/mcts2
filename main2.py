# main.py
import time
import random
import os # Добавлен импорт os для доступа к переменным окружения
from typing import List, Set, Optional, Tuple, Any

# Используем корневой card.py
from card import card_from_str, card_to_str, Card # Card используется для аннотаций
from game_state import GameState # Использует обновленный scoring/board
from mcts_agent import MCTSAgent # Использует обновленный MCTSNode/GameState
from fantasyland_solver import FantasylandSolver # Использует обновленный scoring/card
# Добавим импорт для проверки фола в get_human_fantasyland_placement
from scoring import check_board_foul # Использует обновленный scoring
from board import PlayerBoard # Для аннотаций типов

# --- Функции для ввода пользователя (Консоль) ---

def get_human_action_street1(hand: List[int], board: 'PlayerBoard') -> Tuple[List[Tuple[int, str, int]], List[int]]: # Карты - int
    """Запрашивает у человека размещение 5 карт."""
    print("\nВаша рука (Улица 1):", ", ".join(card_to_str(c) for c in hand))
    print("Ваша доска:")
    print(board) # Использует board.__str__
    placements = []
    placed_card_indices_in_hand = set() # Индексы карт из руки, которые уже размещены
    placed_slots_on_board = set() # Слоты на доске, занятые в этом ходу

    while len(placements) < 5:
        print(f"\n--- Размещение карты {len(placements) + 1}/5 ---")
        # Показываем доступные карты из руки
        available_hand_cards = []
        print("Доступные карты в руке:")
        for idx, card_int in enumerate(hand):
            if idx not in placed_card_indices_in_hand:
                available_hand_cards.append((idx, card_int))
                print(f"  {len(available_hand_cards)}. {card_to_str(card_int)}") # Используем card_to_str

        # Выбор карты
        while True:
            try:
                choice_str = input(f"Выберите номер карты для размещения (1-{len(available_hand_cards)}): ")
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(available_hand_cards):
                    hand_idx, chosen_card_int = available_hand_cards[choice_idx] # Карта - int
                    break
                else: print("Неверный номер карты.")
            except ValueError: print("Неверный ввод. Введите число.")

        # Показываем доступные слоты
        print("Доступные слоты:")
        available_slots = []
        slot_map = {} # Для обратного отображения выбора пользователя в (row, index)
        slot_counter = 1
        for r_name in PlayerBoard.ROW_NAMES:
             for c_idx in range(PlayerBoard.ROW_CAPACITY[r_name]):
                  slot_key = (r_name, c_idx)
                  # Проверяем, что слот пуст на доске И не занят в текущем ходе
                  if board.rows[r_name][c_idx] is None and slot_key not in placed_slots_on_board:
                       available_slots.append(slot_key)
                       print(f"  {slot_counter}. {r_name.upper()}[{c_idx}]")
                       slot_map[slot_counter] = slot_key
                       slot_counter += 1

        # Выбор слота
        while True:
            try:
                slot_choice_str = input(f"Выберите номер слота для {card_to_str(chosen_card_int)} (1-{len(available_slots)}): ")
                slot_choice_num = int(slot_choice_str)
                if slot_choice_num in slot_map:
                    chosen_slot = slot_map[slot_choice_num]
                    row_name, index = chosen_slot
                    break
                else: print("Неверный номер слота.")
            except ValueError: print("Неверный ввод. Введите число.")

        # Добавляем размещение (карта - int)
        placements.append((chosen_card_int, row_name, index))
        placed_card_indices_in_hand.add(hand_idx)
        placed_slots_on_board.add(chosen_slot)
        print(f"Карта {card_to_str(chosen_card_int)} размещена в {row_name.upper()}[{index}]")

        # Временно отображаем доску с размещенными картами этого хода
        temp_board = board.copy()
        for p_card, p_row, p_idx in placements:
             temp_board.add_card(p_card, p_row, p_idx) # Добавляем int карту
        print("Текущее размещение на этой улице:")
        print(temp_board)


    return placements, [] # Пустой сброс для улицы 1

def get_human_action_pineapple(hand: List[int], board: 'PlayerBoard') -> Optional[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]]: # Карты - int
    """Запрашивает у человека ход Pineapple (улицы 2-5)."""
    print("\nВаша рука (Улицы 2-5):", ", ".join(f"{i+1}:{card_to_str(c)}" for i, c in enumerate(hand)))
    print("Ваша доска:")
    print(board)

    # 1. Выбор карты для сброса
    discarded_card_int = None # Карта - int
    discard_idx = -1
    while discarded_card_int is None:
        try:
            discard_choice_str = input(f"Выберите номер карты для СБРОСА (1, 2 или 3): ")
            discard_idx = int(discard_choice_str) - 1
            if 0 <= discard_idx < 3:
                discarded_card_int = hand[discard_idx]
                print(f"Выбрана для сброса: {card_to_str(discarded_card_int)}")
            else: print("Неверный номер карты.")
        except ValueError: print("Неверный ввод. Введите число.")

    cards_to_place = [card for i, card in enumerate(hand) if i != discard_idx] # Список int карт
    print(f"Карты для размещения: {card_to_str(cards_to_place[0])}, {card_to_str(cards_to_place[1])}")

    placements = []
    placed_card_indices_in_hand = set() # Индексы карт из ОРИГИНАЛЬНОЙ руки (0, 1, 2), которые размещены
    placed_slots_on_board = set()

    # 2. Размещение двух карт
    for i in range(2):
        print(f"\n--- Размещение карты {i + 1}/2 ---")
        # Показываем доступные для размещения карты
        available_placement_cards = []
        print("Доступные карты для размещения:")
        original_hand_map = {} # card_int -> original_index
        for card_idx, card_int in enumerate(hand):
             # Карта доступна, если это не сброшенная карта и она еще не размещена в этом ходу
             if card_int != discarded_card_int and card_idx not in placed_card_indices_in_hand:
                  available_placement_cards.append(card_int)
                  original_hand_map[card_int] = card_idx
                  print(f"  {len(available_placement_cards)}. {card_to_str(card_int)}")

        # Выбор карты для размещения
        while True:
            try:
                card_choice_str = input(f"Выберите номер карты для размещения (1-{len(available_placement_cards)}): ")
                card_choice_idx = int(card_choice_str) - 1
                if 0 <= card_choice_idx < len(available_placement_cards):
                    chosen_card_int = available_placement_cards[card_choice_idx] # Карта - int
                    original_idx = original_hand_map[chosen_card_int]
                    break
                else: print("Неверный номер карты.")
            except ValueError: print("Неверный ввод. Введите число.")

        # Показываем доступные слоты
        print("Доступные слоты:")
        available_slots = []
        slot_map = {}
        slot_counter = 1
        for r_name in PlayerBoard.ROW_NAMES:
             for c_idx in range(PlayerBoard.ROW_CAPACITY[r_name]):
                  slot_key = (r_name, c_idx)
                  # Слот доступен, если пуст на доске и не занят в этом ходу
                  if board.rows[r_name][c_idx] is None and slot_key not in placed_slots_on_board:
                       available_slots.append(slot_key)
                       print(f"  {slot_counter}. {r_name.upper()}[{c_idx}]")
                       slot_map[slot_counter] = slot_key
                       slot_counter += 1
        if not available_slots:
             print("Ошибка: Нет доступных слотов!")
             return None # Не должно происходить, если get_legal_actions сработал

        # Выбор слота
        while True:
            try:
                slot_choice_str = input(f"Выберите номер слота для {card_to_str(chosen_card_int)} (1-{len(available_slots)}): ")
                slot_choice_num = int(slot_choice_str)
                if slot_choice_num in slot_map:
                    chosen_slot = slot_map[slot_choice_num]
                    row_name, index = chosen_slot
                    break
                else: print("Неверный номер слота.")
            except ValueError: print("Неверный ввод. Введите число.")

        # Добавляем размещение (карта - int)
        placements.append((chosen_card_int, row_name, index))
        placed_card_indices_in_hand.add(original_idx) # Отмечаем исходный индекс размещенной карты
        placed_slots_on_board.add(chosen_slot)
        print(f"Карта {card_to_str(chosen_card_int)} размещена в {row_name.upper()}[{index}]")

        # Временно отображаем доску
        temp_board = board.copy()
        for p_card, p_row, p_idx in placements:
             temp_board.add_card(p_card, p_row, p_idx) # Добавляем int карту
        print("Текущее размещение на этой улице:")
        print(temp_board)

    if len(placements) != 2: return None # Ошибка

    # Формируем кортеж действия (карты - int)
    return (placements[0], placements[1], discarded_card_int)


def get_human_fantasyland_placement(hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]: # Карты - int
     """Запрашивает у человека размещение Фантазии."""
     print("\n--- FANTASYLAND ---")
     print("Ваша рука:", ", ".join(card_to_str(c) for c in hand))
     n_cards = len(hand)
     n_place = 13
     n_discard = n_cards - n_place
     if n_discard < 0:
          print("Ошибка: Недостаточно карт для Фантазии!")
          return None, None

     discarded_cards = [] # Список int карт для сброса
     placed_cards_map = {} # card_int -> (row, index) - для отслеживания размещения
     placement_dict = {'top': [None]*3, 'middle': [None]*5, 'bottom': [None]*5} # Используем слоты

     # 1. Выбор карт для сброса (если нужно)
     if n_discard > 0:
         print(f"\n--- Выбор {n_discard} карт для сброса ---")
         available_for_discard = list(enumerate(hand)) # [(0, card_int), (1, card_int), ...]
         while len(discarded_cards) < n_discard:
             print("Доступные карты для сброса:")
             current_options = []
             for idx, card_int in available_for_discard:
                  # Карта доступна, если она еще не выбрана для сброса
                  if card_int not in discarded_cards:
                       current_options.append((idx, card_int))
                       print(f"  {len(current_options)}. {card_to_str(card_int)} (исходный индекс {idx})")

             while True:
                  try:
                       choice_str = input(f"Выберите номер карты для сброса ({len(discarded_cards)+1}/{n_discard}): ")
                       choice_idx = int(choice_str) - 1
                       if 0 <= choice_idx < len(current_options):
                            _, chosen_card_int = current_options[choice_idx] # Карта - int
                            discarded_cards.append(chosen_card_int)
                            print(f"Карта {card_to_str(chosen_card_int)} выбрана для сброса.")
                            break
                       else: print("Неверный номер карты.")
                  except ValueError: print("Неверный ввод. Введите число.")
         print(f"Выбраны для сброса: {', '.join(card_to_str(c) for c in discarded_cards)}")

     # 2. Размещение оставшихся 13 карт
     cards_to_place = [c for c in hand if c not in discarded_cards] # Список int карт
     print("\n--- Размещение 13 карт ---")
     if len(cards_to_place) != 13:
          print("Ошибка: Неверное количество карт для размещения после сброса!")
          return None, None

     placed_count = 0
     while placed_count < 13:
         print(f"\n--- Размещение карты {placed_count + 1}/13 ---")
         # Показываем доступные для размещения карты
         available_placement_cards = []
         print("Доступные карты для размещения:")
         for card_int in cards_to_place:
              # Карта доступна, если она еще не размещена
              if card_int not in placed_cards_map:
                   available_placement_cards.append(card_int)
                   print(f"  {len(available_placement_cards)}. {card_to_str(card_int)}")

         # Выбор карты
         while True:
            try:
                card_choice_str = input(f"Выберите номер карты для размещения (1-{len(available_placement_cards)}): ")
                card_choice_idx = int(card_choice_str) - 1
                if 0 <= card_choice_idx < len(available_placement_cards):
                    chosen_card_int = available_placement_cards[card_choice_idx] # Карта - int
                    break
                else: print("Неверный номер карты.")
            except ValueError: print("Неверный ввод. Введите число.")

         # Показываем доступные слоты
         print("Доступные слоты:")
         available_slots = []
         slot_map = {}
         slot_counter = 1
         for r_name in PlayerBoard.ROW_NAMES:
              for c_idx in range(PlayerBoard.ROW_CAPACITY[r_name]):
                   # Слот доступен, если он пуст в нашем временном размещении
                   if placement_dict[r_name][c_idx] is None:
                        slot_key = (r_name, c_idx)
                        available_slots.append(slot_key)
                        print(f"  {slot_counter}. {r_name.upper()}[{c_idx}]")
                        slot_map[slot_counter] = slot_key
                        slot_counter += 1

         # Выбор слота
         while True:
            try:
                slot_choice_str = input(f"Выберите номер слота для {card_to_str(chosen_card_int)} (1-{len(available_slots)}): ")
                slot_choice_num = int(slot_choice_str)
                if slot_choice_num in slot_map:
                    chosen_slot = slot_map[slot_choice_num]
                    row_name, index = chosen_slot
                    break
                else: print("Неверный номер слота.")
            except ValueError: print("Неверный ввод. Введите число.")

         # Размещаем карту (int)
         placement_dict[row_name][index] = chosen_card_int
         placed_cards_map[chosen_card_int] = (row_name, index) # Запоминаем размещение
         placed_count += 1
         print(f"Карта {card_to_str(chosen_card_int)} размещена в {row_name.upper()}[{index}]")

         # Отображаем текущее размещение
         print("Текущее размещение Фантазии:")
         temp_board_vis = ""
         for r_name in PlayerBoard.ROW_NAMES:
              row_str = [card_to_str(c) if c is not None else "__" for c in placement_dict[r_name]]
              temp_board_vis += f"{r_name.upper():<6}: " + " ".join(row_str) + "\n"
         print(temp_board_vis.strip())

     # Проверяем валидность размещения (без фола) перед возвратом
     # Собираем финальные списки карт (int) из слотов
     final_placement_lists = {
          'top': [c for c in placement_dict['top'] if c is not None],
          'middle': [c for c in placement_dict['middle'] if c is not None],
          'bottom': [c for c in placement_dict['bottom'] if c is not None]
     }
     # Проверяем количество карт
     if len(final_placement_lists['top']) != 3 or \
        len(final_placement_lists['middle']) != 5 or \
        len(final_placement_lists['bottom']) != 5:
          print("Ошибка: Не все карты размещены корректно!")
          return None, None # Должно быть 13 карт

     # Проверяем фол с помощью обновленной функции scoring
     if not check_board_foul(final_placement_lists['top'], final_placement_lists['middle'], final_placement_lists['bottom']):
         return final_placement_lists, discarded_cards # Возвращаем карты int
     else:
         print("Ошибка: Собранная рука - фол! Попробуйте разместить заново.")
         # В реальной игре нужно дать переделать, здесь возвращаем ошибку
         # Чтобы дать переделать, нужно обернуть все в цикл while
         return None, None


def play_game():
    """Основной цикл игры в консоли."""
    num_players = 2
    human_player_idx = 0 # 0 или 1, или None для AI vs AI
    # Параметры AI можно задать здесь или через переменные окружения
    ai_time_limit = int(os.environ.get('MCTS_TIME_LIMIT_MS', 5000))
    ai_rave_k = int(os.environ.get('MCTS_RAVE_K', 500))
    ai_workers = int(os.environ.get('NUM_WORKERS', MCTSAgent.DEFAULT_NUM_WORKERS))
    ai_rollouts_leaf = int(os.environ.get('ROLLOUTS_PER_LEAF', MCTSAgent.DEFAULT_ROLLOUTS_PER_LEAF))

    ai_player = MCTSAgent(time_limit_ms=ai_time_limit,
                          rave_k=ai_rave_k,
                          num_workers=ai_workers,
                          rollouts_per_leaf=ai_rollouts_leaf)
    # fl_solver не нужен отдельно, т.к. используется внутри MCTSAgent

    game_score = [0, 0]
    dealer_idx = random.choice([0, 1])
    # Сохраняем статус ФЛ и кол-во карт между раундами
    fantasyland_status_carryover = [False] * num_players
    fantasyland_cards_carryover = [0] * num_players

    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'='*10} РАУНД {round_num} {'='*10}")
        dealer_idx = 1 - dealer_idx # Меняем дилера
        print(f"Дилер: Игрок {dealer_idx}")

        # Инициализация состояния раунда
        current_state = GameState(dealer_idx=dealer_idx,
                                  fantasyland_status=list(fantasyland_status_carryover),
                                  fantasyland_cards_to_deal=list(fantasyland_cards_carryover))
        current_state.start_new_round(dealer_idx) # Раздает карты ФЛ и/или 1й улицы
        print(f"Статус Фантазии: {current_state.fantasyland_status}")
        if current_state.is_fantasyland_round: print("--- Раунд Фантазии ---")
        else: print("--- Обычный раунд ---")

        # Основной цикл раунда
        while not current_state.is_round_over():
            made_move_in_loop = False
            player_to_act = -1

            # --- Определяем, кто ходит ---
            gs = current_state
            if gs.is_fantasyland_round:
                 # Приоритет у ФЛ игроков с картами
                 for i in range(gs.NUM_PLAYERS):
                      if gs.fantasyland_status[i] and not gs._player_finished_round[i] and gs.fantasyland_hands[i] is not None:
                           player_to_act = i; break
                 # Затем не-ФЛ игроки с картами
                 if player_to_act == -1:
                      for i in range(gs.NUM_PLAYERS):
                           if not gs.fantasyland_status[i] and not gs._player_finished_round[i] and gs.current_hands.get(i) is not None:
                                player_to_act = i; break
            else: # Обычный раунд
                 p_idx = gs.current_player_idx
                 if not gs._player_finished_round[p_idx] and gs.current_hands.get(p_idx) is not None:
                      player_to_act = p_idx

            # Если никто не может ходить (ждут карт или все закончили)
            if player_to_act == -1 and not current_state.is_round_over():
                 print("Никто не может ходить, проверка раздачи...")
                 # --- Логика раздачи карт ---
                 needs_dealing = False
                 # Переход улицы в обычном раунде (если ОБА сходили на текущей)
                 if not current_state.is_fantasyland_round and all(current_state._player_acted_this_street):
                      current_state.street += 1
                      if current_state.street <= 5:
                           print(f"\n--- Переход на Улицу {current_state.street} ---")
                           current_state._player_acted_this_street = [False] * num_players
                           current_state.current_player_idx = 1 - current_state.dealer_idx
                           needs_dealing = True
                 # Передача хода в обычном раунде (если один сходил, а другой нет)
                 elif not current_state.is_fantasyland_round and not all(current_state._player_acted_this_street):
                      current_player = current_state.current_player_idx
                      other_player = 1 - current_player
                      if current_state._player_acted_this_street[current_player] and \
                         not current_state._player_acted_this_street[other_player] and \
                         not current_state._player_finished_round[other_player]:
                           current_state.current_player_idx = other_player
                           needs_dealing = current_state.current_hands.get(other_player) is None

                 # Раздача карт (если нужно)
                 players_to_deal = []
                 if needs_dealing and not current_state.is_fantasyland_round:
                      p_idx = current_state.current_player_idx
                      if not current_state._player_finished_round[p_idx] and current_state.current_hands.get(p_idx) is None:
                           players_to_deal.append(p_idx)
                 elif current_state.is_fantasyland_round:
                      for p_idx_deal in range(num_players):
                           if not current_state.fantasyland_status[p_idx_deal] and \
                              not current_state._player_finished_round[p_idx_deal] and \
                              current_state.current_hands.get(p_idx_deal) is None:
                                   players_to_deal.append(p_idx_deal)

                 if players_to_deal:
                      print(f"Раздача карт игрокам: {players_to_deal}")
                      for p_idx_deal in players_to_deal:
                           current_state._deal_street_to_player(p_idx_deal)
                      made_move_in_loop = True # Считаем раздачу "ходом" для продолжения цикла
                 else:
                      # Если раздавать некому, но раунд не закончен, что-то не так
                      print("Предупреждение: Никто не ходит, раздавать некому. Проверка состояния...")
                      if all(p is None for p in current_state.current_hands.values()) and \
                         all(p is None for p in current_state.fantasyland_hands):
                           print("Ошибка: У всех активных игроков нет карт. Прерывание раунда.")
                           break # Прерываем, чтобы избежать зацикливания
                 continue # Переходим к следующей итерации после раздачи

            # --- Ход игрока ---
            p_idx = player_to_act
            action = None
            is_fantasyland_turn = current_state.is_fantasyland_round and current_state.fantasyland_status[p_idx]

            print(f"\n--- Ход Игрока {p_idx} ---")
            if is_fantasyland_turn:
                 print("(Фантазия)")
                 hand = current_state.fantasyland_hands[p_idx] # Список int
                 if p_idx == human_player_idx:
                      placement, discarded = None, None
                      for attempt in range(3): # Даем 3 попытки
                           placement, discarded = get_human_fantasyland_placement(hand)
                           if placement: break
                           print("Попробуйте разместить снова.")
                      if placement:
                           # Формируем действие для apply_fantasyland_placement
                           action = ("FANTASYLAND_PLACEMENT", placement, discarded)
                      else:
                           print("Не удалось разместить Фантазию без фола. Рука будет считаться фолом.")
                           action = ("FANTASYLAND_FOUL", hand) # Передаем руку для сброса
                 else: # AI
                      print(f"AI Игрок {p_idx} решает Фантазию...")
                      action = ai_player.choose_action(current_state) # Делегирует солверу
            else: # Обычный ход
                 print(f"(Улица {current_state.street})")
                 hand = current_state.current_hands[p_idx] # Список int
                 board = current_state.boards[p_idx]
                 if p_idx == human_player_idx:
                      if current_state.street == 1:
                           action = get_human_action_street1(hand, board)
                      else:
                           action = get_human_action_pineapple(hand, board)
                 else: # AI
                      print(f"AI Игрок {p_idx} думает...")
                      action = ai_player.choose_action(current_state) # MCTS
                      print(f"AI выбрал: {ai_player._format_action(action)}")

            # --- Применение действия ---
            if action is None and p_idx == human_player_idx:
                 print("Отмена хода или ошибка ввода. Попробуйте снова.")
                 continue # Возвращаемся к запросу хода человека

            if action is None and p_idx != human_player_idx:
                 # Если AI не смог выбрать действие (маловероятно)
                 print(f"Ошибка: AI Игрок {p_idx} не смог выбрать ход. Считаем фолом.")
                 if is_fantasyland_turn:
                      current_state = current_state.apply_fantasyland_foul(p_idx, hand)
                 else:
                      current_state.boards[p_idx].is_foul = True
                      current_state._player_finished_round[p_idx] = True
                      if current_state.current_hands.get(p_idx):
                           current_state.private_discard[p_idx].extend(current_state.current_hands[p_idx])
                           current_state.current_hands[p_idx] = None
                 made_move_in_loop = True
            else:
                 # Применяем действие
                 if isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT":
                      _, placement, discarded = action
                      current_state = current_state.apply_fantasyland_placement(p_idx, placement, discarded)
                 elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL":
                      _, hand_to_discard = action
                      current_state = current_state.apply_fantasyland_foul(p_idx, hand_to_discard)
                 elif isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT":
                      # Это действие от get_legal_actions для AI, его нужно передать в choose_action
                      # Здесь оно не должно применяться напрямую
                      print(f"Ошибка: Попытка применить FANTASYLAND_INPUT для игрока {p_idx}")
                      continue
                 else: # Обычное действие
                      current_state = current_state.apply_action(p_idx, action)
                 made_move_in_loop = True

            # Отображаем доску после хода
            print(f"Доска Игрока {p_idx} после хода:")
            print(current_state.boards[p_idx])
            if current_state._player_finished_round[p_idx]:
                 print(f"Игрок {p_idx} завершил раунд.")


        # --- Подсчет очков раунда ---
        if current_state.is_round_over():
             print("\n--- Раунд Завершен ---")
             print("Финальные доски:")
             print("Игрок 0:")
             print(current_state.boards[0])
             print("Игрок 1:")
             print(current_state.boards[1])

             score_diff = current_state.get_terminal_score() # Использует обновленный scoring
             print(f"Счет за раунд (P0 vs P1): {score_diff}")
             game_score[0] += score_diff
             game_score[1] -= score_diff
             print(f"Общий счет: P0={game_score[0]}, P1={game_score[1]}")

             # Обновляем статус ФЛ для СЛЕДУЮЩЕГО раунда
             fantasyland_status_carryover = list(current_state.next_fantasyland_status)
             fantasyland_cards_carryover = list(current_state.fantasyland_cards_to_deal)
             print(f"Статус Фантазии на след. раунд: {fantasyland_status_carryover} (Карты: {fantasyland_cards_carryover})")
        else:
             print("\n--- Раунд НЕ Завершен (Ошибка?) ---")
             # Если вышли из цикла не по is_round_over, значит была проблема
             fantasyland_status_carryover = [False] * num_players # Сбрасываем ФЛ на всякий случай
             fantasyland_cards_carryover = [0] * num_players

        # Запрос на следующий раунд
        cont = input("Сыграть еще раунд? (y/n): ").lower()
        if cont != 'y':
            break

    print("\n===== ИГРА ОКОНЧЕНА =====")
    print(f"Финальный счет: Игрок 0: {game_score[0]}, Игрок 1: {game_score[1]}")

if __name__ == "__main__":
    # Установка кодировки для Windows, если необходимо
    import sys, io
    if sys.stdout.encoding != 'utf-8':
         try:
              sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
         except Exception as e:
              print(f"Warning: Could not set stdout encoding to utf-8: {e}")
    if sys.stderr.encoding != 'utf-8':
         try:
              sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
         except Exception as e:
              print(f"Warning: Could not set stderr encoding to utf-8: {e}")

    play_game()