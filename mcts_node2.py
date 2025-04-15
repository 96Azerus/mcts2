# mcts_node.py
import math
import random
import traceback
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import Counter

from game_state import GameState # Используем обновленный GameState
# Используем корневой Card и его утилиты
from card import Card, card_to_str, RANK_MAP, STR_RANKS, Card as CardUtils # Используем CardUtils для статических методов
# Используем обновленный scoring для констант и функций
from scoring import (RANK_CLASS_QUADS, RANK_CLASS_TRIPS, get_hand_rank_safe,
                     check_board_foul, get_row_royalty, RANK_CLASS_PAIR,
                     RANK_CLASS_HIGH_CARD)
from itertools import combinations
# Используем FantasylandSolver как есть, он вызывает методы GameState/Board
from fantasyland_solver import FantasylandSolver

class MCTSNode:
    """Узел дерева MCTS для OFC Pineapple с RAVE."""
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None, action: Optional[Any] = None):
        self.game_state: GameState = game_state # Состояние игры в этом узле
        self.parent: Optional['MCTSNode'] = parent # Родительский узел
        self.action: Optional[Any] = action # Действие, которое привело в этот узел
        self.children: Dict[Any, 'MCTSNode'] = {} # Дочерние узлы (действие -> узел)
        # Список действий, которые еще не были развернуты из этого узла
        self.untried_actions: Optional[List[Any]] = None
        self.visits: int = 0 # Количество посещений (симуляций через этот узел)
        self.total_reward: float = 0.0 # Суммарная награда (с точки зрения игрока 0)
        # RAVE - хранит информацию о наградах для действий, встреченных в симуляциях ниже этого узла
        self.rave_visits: Dict[Any, int] = {} # Посещения для RAVE (действие -> количество)
        self.rave_total_reward: Dict[Any, float] = {} # Суммарная награда для RAVE (действие -> награда P0)

    def _get_player_to_move(self) -> int:
         """ Определяет индекс игрока, который должен ходить из текущего состояния. Возвращает -1, если раунд завершен. """
         gs = self.game_state
         if gs.is_round_over(): return -1 # Терминальное состояние

         if gs.is_fantasyland_round:
              # В ФЛ раунде приоритет у игроков в ФЛ, если у них есть карты
              for i in range(gs.NUM_PLAYERS):
                   if gs.fantasyland_status[i] and not gs._player_finished_round[i] and gs.fantasyland_hands[i] is not None:
                        return i
              # Затем ходят не-ФЛ игроки, если у них есть карты
              for i in range(gs.NUM_PLAYERS):
                   if not gs.fantasyland_status[i] and not gs._player_finished_round[i] and gs.current_hands.get(i) is not None:
                        return i
              # Если никто не может ходить (все ждут карт или закончили), возвращаем -1
              return -1

         else: # Обычный раунд
              p_idx = gs.current_player_idx
              # Ходит текущий игрок, если он не закончил и у него есть карты
              if not gs._player_finished_round[p_idx] and gs.current_hands.get(p_idx) is not None:
                   return p_idx
              else:
                   # Если текущий не может ходить, проверяем другого
                   other_player = 1 - p_idx
                   if not gs._player_finished_round[other_player] and gs.current_hands.get(other_player) is not None:
                        # ВАЖНО: В MCTS мы не меняем current_player_idx, это делает apply_action.
                        # Здесь мы просто определяем, КТО БЫ ходил следующим, если бы игра продолжалась.
                        # Но для выбора действия из узла нам нужен тот, кто ходит СЕЙЧАС.
                        # Если текущий игрок не может ходить (нет карт или закончил), то из этого узла ходить некому.
                        return -1
                   else:
                        # Если и другой не может, то ходить некому
                        return -1

    def expand(self) -> Optional['MCTSNode']:
        """
        Расширяет узел, выбирая одно неиспробованное действие,
        применяя его и создавая новый дочерний узел.
        Возвращает новый дочерний узел или None, если расширение невозможно.
        """
        player_to_move = self._get_player_to_move()
        if player_to_move == -1: return None # Нельзя расширить терминальный узел или узел, где никто не ходит

        # Инициализируем неиспробованные действия, если нужно
        if self.untried_actions is None:
             # get_legal_actions_for_player вернет [] если у игрока нет карт
             self.untried_actions = self.game_state.get_legal_actions_for_player(player_to_move)
             random.shuffle(self.untried_actions)
             # Инициализация RAVE для новых действий
             for act in self.untried_actions:
                 if act not in self.rave_visits:
                     self.rave_visits[act] = 0
                     self.rave_total_reward[act] = 0.0

        # Если больше нет неиспробованных действий
        if not self.untried_actions:
            # print(f"Warning: No untried actions left to expand for player {player_to_move} in state {self.game_state.get_state_representation()}")
            return None

        action = self.untried_actions.pop()
        next_state = None

        # Применяем действие (кроме ФЛ, т.к. там солвер)
        # Проверка на ФЛ здесь избыточна, т.к. get_legal_actions вернет спец. действие
        if isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT":
             # Это действие не должно применяться здесь, оно для агента
             print(f"Error: Attempted to expand Fantasyland input action.")
             return None
        else: # Обычное действие
             try:
                 # apply_action возвращает НОВОЕ состояние
                 next_state = self.game_state.apply_action(player_to_move, action)
                 if next_state is self.game_state: # Если apply_action вернул старое состояние (ошибка)
                      print(f"Error: apply_action returned the same state for player {player_to_move}. Action: {action}")
                      return None
             except Exception as e:
                 print(f"Error applying action during expand for player {player_to_move}: {e}")
                 traceback.print_exc()
                 return None # Не удалось применить действие

        if next_state is None: return None # Если apply_action вернул None

        # Создаем дочерний узел
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def is_terminal(self) -> bool:
        """Проверяет, является ли состояние в узле терминальным."""
        return self.game_state.is_round_over()

    def rollout(self, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
        """
        Проводит симуляцию (rollout) до конца игры из текущего узла,
        используя эвристическую политику.
        Возвращает финальный счет с точки зрения perspective_player и набор сделанных действий.
        """
        current_rollout_state = self.game_state.copy()
        simulation_actions_set = set() # Собираем действия, сделанные в этой симуляции для RAVE
        MAX_ROLLOUT_STEPS = 50 # Ограничение на случай зацикливания
        steps = 0

        while not current_rollout_state.is_round_over() and steps < MAX_ROLLOUT_STEPS:
            steps += 1
            made_move_this_iter = False
            player_acted_in_iter = -1

            # Определяем, кто ходит в симуляции
            player_to_act_rollout = -1
            gs_rollout = current_rollout_state
            if gs_rollout.is_fantasyland_round:
                 # Приоритет у ФЛ игроков с картами
                 for i in range(gs_rollout.NUM_PLAYERS):
                      if gs_rollout.fantasyland_status[i] and not gs_rollout._player_finished_round[i] and gs_rollout.fantasyland_hands[i] is not None:
                           player_to_act_rollout = i; break
                 # Затем не-ФЛ игроки с картами
                 if player_to_act_rollout == -1:
                      for i in range(gs_rollout.NUM_PLAYERS):
                           if not gs_rollout.fantasyland_status[i] and not gs_rollout._player_finished_round[i] and gs_rollout.current_hands.get(i) is not None:
                                player_to_act_rollout = i; break
            else: # Обычный раунд
                 p_idx = gs_rollout.current_player_idx
                 if not gs_rollout._player_finished_round[p_idx] and gs_rollout.current_hands.get(p_idx) is not None:
                      player_to_act_rollout = p_idx

            # Если есть игрок для хода
            if player_to_act_rollout != -1:
                action = None
                is_fl_placement = current_rollout_state.is_fantasyland_round and current_rollout_state.fantasyland_status[player_to_act_rollout]

                if is_fl_placement:
                    hand = current_rollout_state.fantasyland_hands[player_to_act_rollout]
                    if hand:
                        # Используем быструю эвристику для ФЛ в роллауте
                        placement, discarded = self._heuristic_fantasyland_placement(hand)
                        if placement:
                            # Применяем размещение ФЛ
                            current_rollout_state = current_rollout_state.apply_fantasyland_placement(player_to_act_rollout, placement, discarded)
                            # Добавляем "мета-действие" ФЛ в набор для RAVE (опционально)
                            # simulation_actions_set.add(("FANTASYLAND_SUCCESS", player_to_act_rollout))
                        else: # Если эвристика не смогла, фолим
                            current_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand)
                            # simulation_actions_set.add(("FANTASYLAND_FAIL", player_to_act_rollout))
                        made_move_this_iter = True
                        player_acted_in_iter = player_to_act_rollout
                else: # Обычный ход
                    hand = current_rollout_state.current_hands.get(player_to_act_rollout)
                    if hand:
                        possible_moves = current_rollout_state.get_legal_actions_for_player(player_to_act_rollout)
                        if possible_moves:
                            # Используем эвристику для выбора хода
                            action = self._heuristic_rollout_policy(current_rollout_state, player_to_act_rollout, possible_moves)
                            if action:
                                simulation_actions_set.add(action) # Добавляем действие в набор для RAVE
                                current_rollout_state = current_rollout_state.apply_action(player_to_act_rollout, action)
                                made_move_this_iter = True
                                player_acted_in_iter = player_to_act_rollout
                            else: # Если эвристика не выбрала ход (маловероятно) -> Фол
                                current_rollout_state = current_rollout_state.apply_action(player_to_act_rollout, None) # Передаем None для фола
                                made_move_this_iter = True
                                player_acted_in_iter = player_to_act_rollout
                        else: # Нет легальных ходов -> Фол
                             current_rollout_state = current_rollout_state.apply_action(player_to_act_rollout, None) # Передаем None для фола
                             made_move_this_iter = True
                             player_acted_in_iter = player_to_act_rollout

            # --- Логика после хода / Раздача карт ---
            # (Эта часть важна для корректного течения симуляции)
            if not current_rollout_state.is_round_over():
                 needs_dealing = False
                 # Переход улицы в обычном раунде (если ОБА сходили на текущей)
                 if not current_rollout_state.is_fantasyland_round and all(current_rollout_state._player_acted_this_street):
                      current_rollout_state.street += 1
                      if current_rollout_state.street <= 5:
                           current_rollout_state._player_acted_this_street = [False] * current_rollout_state.NUM_PLAYERS
                           current_rollout_state.current_player_idx = 1 - current_rollout_state.dealer_idx
                           needs_dealing = True
                 # Передача хода в обычном раунде (если один сходил, а другой нет)
                 elif not current_rollout_state.is_fantasyland_round and made_move_this_iter and player_acted_in_iter != -1:
                      current_player = player_acted_in_iter # Кто только что сходил
                      other_player = 1 - current_player
                      # Если другой игрок еще не ходил на этой улице и не закончил раунд
                      if not current_rollout_state._player_acted_this_street[other_player] and \
                         not current_rollout_state._player_finished_round[other_player]:
                           current_rollout_state.current_player_idx = other_player
                           # Раздаем другому игроку, если у него нет карт
                           needs_dealing = current_rollout_state.current_hands.get(other_player) is None

                 # Раздача карт (если нужно)
                 players_to_deal = []
                 if needs_dealing and not current_rollout_state.is_fantasyland_round:
                      # Раздаем текущему игроку (который должен ходить)
                      p_idx = current_rollout_state.current_player_idx
                      if not current_rollout_state._player_finished_round[p_idx] and current_rollout_state.current_hands.get(p_idx) is None:
                           players_to_deal.append(p_idx)
                 elif current_rollout_state.is_fantasyland_round:
                      # В ФЛ раунде раздаем всем не-ФЛ игрокам, у которых нет карт
                      for p_idx_deal in range(current_rollout_state.NUM_PLAYERS):
                           if not current_rollout_state.fantasyland_status[p_idx_deal] and \
                              not current_rollout_state._player_finished_round[p_idx_deal] and \
                              current_rollout_state.current_hands.get(p_idx_deal) is None:
                                   players_to_deal.append(p_idx_deal)

                 # Раздаем карты
                 for p_idx_deal in players_to_deal:
                      current_rollout_state._deal_street_to_player(p_idx_deal)

            # Защита от зацикливания, если никто не смог сходить (например, все ждут карт)
            if not current_rollout_state.is_round_over() and not made_move_this_iter and steps > 1:
                 # print("Warning: Rollout stuck? No move made.")
                 break # Прерываем симуляцию

        # Раунд завершен (или превышен лимит шагов)
        if steps >= MAX_ROLLOUT_STEPS:
             # print("Warning: Rollout reached max steps.")
             pass # Считаем очки как есть

        # Получаем финальный счет с точки зрения игрока 0
        final_score_p0 = current_rollout_state.get_terminal_score() # Использует обновленный scoring

        # Возвращаем счет с нужной точки зрения и набор действий
        if perspective_player == 0:
            return float(final_score_p0), simulation_actions_set
        elif perspective_player == 1:
            return float(-final_score_p0), simulation_actions_set
        else: # Нейтральная перспектива (не используется)
            return 0.0, simulation_actions_set

    def _heuristic_rollout_policy(self, state: GameState, player_idx: int, actions: List[Any]) -> Optional[Any]:
        """Улучшенная эвристика для выбора хода в симуляции, использует новые оценки."""
        if not actions: return None

        # Улица 1: Размещение 5 карт
        if state.street == 1:
            best_action = None
            best_score = -float('inf')
            # Ограничиваем количество проверяемых действий
            num_actions_to_check = min(len(actions), 20) # Можно настроить
            actions_sample = random.sample(actions, num_actions_to_check)

            for action in actions_sample:
                placements, _ = action # action = ([(card_int, row, idx), ...], [])
                score = 0
                temp_board = state.boards[player_idx].copy()
                valid = True
                # Применяем размещение на временную доску
                for card_int, row, index in placements:
                     if not temp_board.add_card(card_int, row, index):
                         valid = False; break
                if not valid: continue # Пропускаем невалидное размещение

                # Оцениваем полученную частичную доску
                # 1. Потенциальные роялти (очень грубо)
                score += temp_board.get_total_royalty() * 0.1 # Малый вес

                # 2. Средний ранг карт в рядах (предпочитаем сильные карты внизу)
                for r_name in temp_board.ROW_NAMES:
                     row_cards = temp_board.get_row_cards(r_name) # Список int
                     if not row_cards: continue
                     # Используем RANK_MAP и CardUtils.get_rank_int из корневого card.py
                     try:
                          rank_sum = sum(RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], 0) for c in row_cards)
                          avg_rank = rank_sum / len(row_cards)
                          if r_name == 'top': score += avg_rank * 0.5
                          elif r_name == 'middle': score += avg_rank * 0.8
                          else: score += avg_rank * 1.0
                          # Штраф за слишком сильные карты наверху/середине
                          if r_name == 'top' and avg_rank > RANK_MAP['9']: score -= (avg_rank - RANK_MAP['9']) * 2
                          if r_name == 'middle' and avg_rank > RANK_MAP['J']: score -= (avg_rank - RANK_MAP['J'])
                     except (IndexError, KeyError, TypeError) as e:
                          print(f"Warning: Error calculating avg rank in rollout policy: {e}")
                          continue # Пропускаем оценку этого ряда

                # 3. Потенциальный фол (очень грубо)
                # Сравниваем ранги рядов, если они заполнены достаточно для оценки
                rank_t = temp_board._get_rank('top') # Использует обновленный scoring
                rank_m = temp_board._get_rank('middle')
                rank_b = temp_board._get_rank('bottom')
                # Штрафуем, если порядок сильно нарушен (используем числовые ранги)
                # get_hand_rank_safe возвращает большие числа для неполных рук
                if rank_m < rank_b - 500: score -= 20 # Middle сильнее Bottom?
                if rank_t < rank_m - 500: score -= 20 # Top сильнее Middle?

                # Добавляем немного случайности
                score += random.uniform(-0.1, 0.1)

                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action if best_action else random.choice(actions)

        # Улицы 2-5: Размещение 2 карт, сброс 1
        else:
            hand = state.current_hands.get(player_idx) # Список int
            if not hand or len(hand) != 3: return random.choice(actions) # На всякий случай

            best_action = None
            best_score = -float('inf')
            current_board = state.boards[player_idx]
            num_actions_to_check = min(len(actions), 30) # Можно настроить
            actions_sample = random.sample(actions, num_actions_to_check)

            for action in actions_sample:
                # action = ((card1_int, r1, i1), (card2_int, r2, i2), discard_int)
                place1, place2, discarded_int = action
                card1_int, row1, idx1 = place1
                card2_int, row2, idx2 = place2
                score = 0

                # 1. Штраф за сброс сильной карты
                try:
                     discard_rank = RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(discarded_int)], 0)
                     score -= discard_rank * 0.5 # Меньше штраф, чем раньше
                except (IndexError, KeyError, TypeError): pass # Игнорируем ошибку, если карта некорректна

                # 2. Оценка каждого размещения
                def placement_score(card_int, row, index, board):
                    b = 0
                    temp_board_eval = board.copy()
                    if not temp_board_eval.add_card(card_int, row, index): return -1000

                    current_row_cards = temp_board_eval.get_row_cards(row) # Список int
                    try:
                         card_rank = RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(card_int)], 0)
                    except (IndexError, KeyError, TypeError): card_rank = 0

                    # Бонус за создание пар/сетов
                    try:
                         rank_counts = Counter(RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], -1) for c in current_row_cards)
                         card_rank_count = rank_counts.get(card_rank, 0)
                         if card_rank_count == 2: b += 5 # Сделали пару
                         if card_rank_count == 3: b += 15 # Сделали сет
                         if card_rank_count == 4: b += 30 # Сделали каре
                    except (IndexError, KeyError, TypeError): pass

                    # Бонус за флеш-дро (очень грубо)
                    try:
                         suits = {CardUtils.get_suit_int(c) for c in current_row_cards}
                         if len(suits) == 1 and len(current_row_cards) >= 3:
                             b += len(current_row_cards) # Бонус за каждую карту во флеше/дро
                    except Exception: pass

                    # Позиционные бонусы/штрафы
                    if row == 'top':
                         if card_rank >= RANK_MAP['Q']: b += 10 # QQ+ хорошо наверху
                         if card_rank_count == 2 and card_rank >= RANK_MAP['6']: b += 5 # Пара 66+
                         if card_rank_count == 3: b += 15 # Трипс наверху - отлично
                         if card_rank < RANK_MAP['6']: b -= 5 # Мелкая карта наверху - плохо
                    elif row == 'middle':
                         if card_rank < RANK_MAP['5']: b -= 3 # Мелкая карта в середине - не очень

                    # Проверка потенциального фола
                    rank_t = temp_board_eval._get_rank('top')
                    rank_m = temp_board_eval._get_rank('middle')
                    rank_b = temp_board_eval._get_rank('bottom')
                    if rank_m < rank_b - 500: b -= 10
                    if rank_t < rank_m - 500: b -= 10

                    return b

                # Оцениваем оба размещения последовательно
                temp_board1 = current_board.copy()
                if not temp_board1.add_card(card1_int, row1, idx1): continue # Пропускаем, если первое размещение не удалось
                score1 = placement_score(card1_int, row1, idx1, current_board)
                score2 = placement_score(card2_int, row2, idx2, temp_board1) # Оцениваем второе на доске ПОСЛЕ первого

                score += score1 + score2
                score += random.uniform(-0.1, 0.1) # Случайность

                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action if best_action else random.choice(actions)

    def _heuristic_fantasyland_placement(self, hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]: # Карты - int
        """Быстрая эвристика для ФЛ в симуляции, использует FantasylandSolver."""
        solver = FantasylandSolver() # Использует обновленный solver
        n_cards = len(hand)
        n_place = 13
        if n_cards < n_place: return None, None

        try:
            # Простая эвристика для сброса: самые младшие карты
            # Используем RANK_MAP и CardUtils.get_rank_int из корневого card.py
            sorted_hand = sorted(hand, key=lambda c: RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], 0))
            n_discard = n_cards - n_place
            discarded_list = sorted_hand[:n_discard] # Список int
            remaining = sorted_hand[n_discard:] # Список int

            if len(remaining) != 13: return None, None # Ошибка

            # Используем простую эвристику размещения из солвера
            placement = solver._try_maximize_royalty_heuristic(remaining) # Работает с int

            # Если простая эвристика не сработала, пытаемся сделать хоть что-то без фола
            if not placement:
                 # Просто сортируем и раскладываем (может быть фол)
                 sorted_remaining = sorted(remaining, key=lambda c: RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], 0), reverse=True)
                 placement = {'bottom': sorted_remaining[0:5], 'middle': sorted_remaining[5:10], 'top': sorted_remaining[10:13]}
                 # Проверяем на фол с помощью обновленной функции scoring
                 if check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                      # print("Warning: Heuristic FL placement resulted in foul.")
                      return None, discarded_list # Возвращаем None, чтобы сигнализировать о фоле

            return placement, discarded_list # Возвращаем карты int

        except Exception as e:
            print(f"Error in heuristic Fantasyland placement: {e}")
            traceback.print_exc()
            # В случае ошибки возвращаем None, что приведет к фолу в роллауте
            default_discard = hand[13:] if len(hand) > 13 else []
            return None, default_discard


    def get_q_value(self, perspective_player: int) -> float:
        """ Возвращает Q-значение узла с точки зрения указанного игрока. """
        if self.visits == 0: return 0.0
        # Определяем, чей ход привел в ЭТОТ узел
        player_who_acted = self.parent._get_player_to_move() if self.parent else -1
        # total_reward хранится с точки зрения игрока 0
        raw_q = self.total_reward / self.visits
        # Корректируем награду в зависимости от того, кто ходил и чья перспектива
        if player_who_acted == perspective_player:
             # Если ходил игрок, с чьей точки зрения смотрим, награда прямая
             return raw_q
        elif player_who_acted != -1:
             # Если ходил оппонент, инвертируем награду
             return -raw_q
        else:
             # Для корневого узла (никто не ходил), возвращаем как есть (с точки зрения P0)
             # Если перспектива P1, нужно инвертировать
             return raw_q if perspective_player == 0 else -raw_q

    def get_rave_q_value(self, action: Any, perspective_player: int) -> float:
         """ Возвращает RAVE Q-значение для действия с точки зрения указанного игрока. """
         rave_visits = self.rave_visits.get(action, 0)
         if rave_visits == 0: return 0.0 # Возвращаем 0, если RAVE-статистики нет
         rave_reward = self.rave_total_reward.get(action, 0.0)
         # rave_total_reward хранится с точки зрения игрока 0
         raw_rave_q = rave_reward / rave_visits
         # Определяем, чей ход БУДЕТ сделан из ТЕКУЩЕГО узла этим действием
         player_to_move = self._get_player_to_move()
         if player_to_move == -1: return 0.0 # Не должно вызываться для терминального
         # Корректируем RAVE награду для перспективы
         if player_to_move == perspective_player:
              # Если будет ходить игрок, с чьей точки зрения смотрим
              return raw_rave_q
         else:
              # Если будет ходить оппонент
              return -raw_rave_q

    def uct_select_child(self, exploration_constant: float, rave_k: float) -> Optional['MCTSNode']:
        """ Выбирает дочерний узел с использованием UCT и RAVE (если применимо). """
        best_score = -float('inf')
        best_child = None
        # Определяем, чей ход из текущего узла
        current_player_perspective = self._get_player_to_move()
        if current_player_perspective == -1: return None # Терминальный узел

        # Используем логарифм от посещений родителя + 1 для избежания log(0)
        parent_visits_log = math.log(self.visits + 1e-6) # Добавляем малое число для стабильности

        children_items = list(self.children.items())
        if not children_items: return None # Нет дочерних узлов

        for action, child in children_items:
            child_visits = child.visits
            rave_visits = self.rave_visits.get(action, 0)
            score = -float('inf')

            if child_visits == 0:
                # Если узел не посещался, используем RAVE (если есть) или бесконечность
                if rave_visits > 0 and rave_k > 0:
                    # Используем RAVE Q-value как оценку для неисследованных узлов
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    # Добавляем exploration бонус, основанный на RAVE посещениях
                    explore_rave = exploration_constant * math.sqrt(parent_visits_log / (rave_visits + 1e-6))
                    score = rave_q + explore_rave
                else:
                    # Отдаем предпочтение неисследованным узлам
                    score = float('inf') # Очень высокий балл для выбора
            else:
                # Стандартный UCB1
                q_child = child.get_q_value(current_player_perspective)
                exploit_term = q_child
                explore_term = exploration_constant * math.sqrt(parent_visits_log / child_visits)
                ucb1_score = exploit_term + explore_term

                # Комбинируем UCB1 и RAVE с использованием параметра rave_k
                if rave_visits > 0 and rave_k > 0:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    # Формула бета для RAVE (из статьи по MCTS)
                    # Используем self.visits (посещения родителя) вместо N(s)
                    beta = math.sqrt(rave_k / (3 * self.visits + rave_k)) if self.visits > 0 else 1.0
                    score = (1 - beta) * ucb1_score + beta * rave_q
                else:
                    # Если RAVE отключен или нет данных, используем чистый UCB1
                    score = ucb1_score

            # Обновляем лучший узел
            if score > best_score:
                best_score = score
                best_child = child
            # Если счет одинаковый, выбираем случайно (кроме случая inf)
            elif score == best_score and score != float('inf') and score != -float('inf'):
                 if random.choice([True, False]):
                      best_child = child

        # Если все узлы имели score = -inf (маловероятно) или не удалось выбрать
        if best_child is None and children_items:
             # print("Warning: Could not determine best child via UCT/RAVE, choosing randomly.")
             best_child = random.choice([child for _, child in children_items])

        return best_child

    def __repr__(self):
        player_idx = self._get_player_to_move()
        player = f'P{player_idx}' if player_idx != -1 else 'T' # T for Terminal
        q_val_p0 = self.get_q_value(0) # Показываем Q с точки зрения P0 для консистентности
        # Пытаемся показать часть действия для идентификации
        action_repr = "Root"
        if self.action:
             try:
                  if isinstance(self.action, tuple) and len(self.action) > 0:
                       # Показываем первый элемент действия (карту или тип)
                       first_elem = self.action[0]
                       if isinstance(first_elem, tuple) and len(first_elem) > 0 and isinstance(first_elem[0], int):
                            action_repr = card_to_str(first_elem[0]) # Первая карта в размещении
                       elif isinstance(first_elem, list) and len(first_elem) > 0 and isinstance(first_elem[0], tuple):
                            action_repr = card_to_str(first_elem[0][0]) # Первая карта в списке размещений
                       elif isinstance(first_elem, str):
                            action_repr = first_elem # Тип действия (ФЛ)
                       else: action_repr = str(self.action)[:10] # Обрезаем
                  else: action_repr = str(self.action)[:10]
             except: action_repr = "???"

        return f"[{player} Act:{action_repr} V={self.visits} Q0={q_val_p0:.2f} N_Child={len(self.children)} U_Act={len(self.untried_actions or [])}]"