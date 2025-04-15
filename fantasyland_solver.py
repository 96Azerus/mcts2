# fantasyland_solver.py
"""
Эвристический солвер для размещения 13 из N (14-17) карт в Фантазии.
Приоритеты: 1. Удержание ФЛ. 2. Максимизация роялти. 3. Не фол.
"""
import random
from typing import List, Tuple, Dict, Optional
from itertools import combinations, permutations
from collections import Counter

# Используем корневой Card и его утилиты
from card import Card, card_to_str, RANK_MAP, STR_RANKS, Card as CardUtils # Импортируем CardUtils для статических методов

# Используем обновленный scoring
from scoring import (check_fantasyland_stay, get_row_royalty, check_board_foul,
                     get_hand_rank_safe, RANK_CLASS_QUADS, RANK_CLASS_TRIPS,
                     RANK_CLASS_HIGH_CARD)

class FantasylandSolver:

    def solve(self, hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]: # Типы карт - int
        """
        Принимает N карт (14-17) в виде int, возвращает лучшее размещение 13 карт и список сброшенных (тоже int).
        Возвращает (None, None) если не найдено валидных размещений.
        """
        n_cards = len(hand)
        n_place = 13
        if n_cards < n_place:
            print(f"Error in FL Solver: Hand size {n_cards} is less than 13.")
            return None, None
        n_discard = n_cards - n_place

        best_overall_placement = None
        best_overall_discarded = None
        best_overall_score = -2 # -1 фол, 0 не ФЛ, 1 ФЛ
        best_overall_royalty = -1
        max_discard_combinations = 50 # Ограничение для производительности

        # Генерируем комбинации карт для сброса
        discard_combinations_list = list(combinations(hand, n_discard))

        # Если комбинаций слишком много, выбираем подмножество
        if len(discard_combinations_list) > max_discard_combinations:
            # Добавляем "умный" сброс (самые младшие карты)
            # Используем RANK_MAP и CardUtils.get_rank_int из корневого card.py
            try:
                 sorted_hand = sorted(hand, key=lambda c: RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], 0))
                 smart_discards = [tuple(sorted_hand[:n_discard])]
            except (IndexError, KeyError, TypeError) as e:
                 print(f"Warning: Error sorting hand for smart discard in FL solver: {e}. Hand: {hand}")
                 smart_discards = [] # Не используем умный сброс при ошибке

            # Добавляем случайные комбинации
            num_random_needed = max(0, max_discard_combinations - len(smart_discards))
            if num_random_needed > 0 and len(discard_combinations_list) > num_random_needed:
                 try:
                      random_discards = random.sample(discard_combinations_list, num_random_needed)
                      combinations_to_check = smart_discards + random_discards
                 except ValueError: # Если smart_discards уже >= max_discard_combinations
                      combinations_to_check = smart_discards[:max_discard_combinations]
            else:
                 combinations_to_check = smart_discards[:max_discard_combinations]

        else:
            combinations_to_check = discard_combinations_list

        # Перебираем варианты сброса
        for discarded_tuple in combinations_to_check:
            discarded_list = list(discarded_tuple)
            remaining_cards = [c for c in hand if c not in discarded_list]
            if len(remaining_cards) != 13: continue # Проверка

            current_best_placement = None
            current_best_score = -1 # -1 фол, 0 не ФЛ, 1 ФЛ
            current_max_royalty = -1
            placements_to_evaluate = []

            # Генерируем несколько эвристических размещений для текущего набора 13 карт
            # Эти методы теперь работают с int картами и используют обновленный scoring
            placement_opt1 = self._try_build_strong_bottom(remaining_cards)
            if placement_opt1: placements_to_evaluate.append(placement_opt1)
            placement_opt2 = self._try_build_set_top(remaining_cards)
            if placement_opt2: placements_to_evaluate.append(placement_opt2)
            placement_opt3 = self._try_maximize_royalty_heuristic(remaining_cards)
            if placement_opt3: placements_to_evaluate.append(placement_opt3)

            # Оцениваем сгенерированные размещения
            for placement in placements_to_evaluate:
                 if not placement: continue # Пропускаем None
                 # _evaluate_placement использует обновленные функции scoring
                 score, royalty = self._evaluate_placement(placement)
                 # Выбираем лучшее размещение для ДАННОГО набора 13 карт
                 # Приоритет: не фол -> удержание ФЛ -> макс роялти
                 if score > current_best_score or \
                    (score == current_best_score and royalty > current_max_royalty):
                     current_best_score = score
                     current_max_royalty = royalty
                     current_best_placement = placement

            # Обновляем лучшее ОБЩЕЕ размещение (среди всех вариантов сброса)
            if current_best_placement:
                 if current_best_score > best_overall_score or \
                    (current_best_score == best_overall_score and current_max_royalty > best_overall_royalty):
                     best_overall_score = current_best_score
                     best_overall_royalty = current_max_royalty
                     best_overall_placement = current_best_placement
                     best_overall_discarded = discarded_list

        # Если не найдено ни одного валидного размещения
        if best_overall_placement is None:
             print("FL Solver Warning: No valid non-foul placement found for any discard combination.")
             # Пытаемся сделать простое размещение без фола для первого варианта сброса
             if discard_combinations_list:
                  first_discard = list(discard_combinations_list[0])
                  first_remaining = [c for c in hand if c not in first_discard]
                  if len(first_remaining) == 13:
                       # Используем RANK_MAP и CardUtils.get_rank_int из корневого card.py
                       try:
                            sorted_remaining = sorted(first_remaining, key=lambda c: RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], 0), reverse=True)
                            simple_placement = {'bottom': sorted_remaining[0:5], 'middle': sorted_remaining[5:10], 'top': sorted_remaining[10:13]}
                            # Проверяем фол с помощью обновленной функции
                            if not check_board_foul(simple_placement['top'], simple_placement['middle'], simple_placement['bottom']):
                                 print("FL Solver: Falling back to simple non-foul placement.")
                                 return simple_placement, first_discard
                       except (IndexError, KeyError, TypeError) as e:
                            print(f"Warning: Error creating fallback placement in FL solver: {e}")
             # Если и это не помогло, возвращаем None
             return None, None

        return best_overall_placement, best_overall_discarded

    def _evaluate_placement(self, placement: Dict[str, List[int]]) -> Tuple[int, int]: # Типы карт - int
        """Оценивает размещение: (-1 фол, 0 не ФЛ, 1 ФЛ), сумма роялти."""
        if not placement or len(placement.get('top', [])) != 3 or len(placement.get('middle', [])) != 5 or len(placement.get('bottom', [])) != 5:
             return -1, -1 # Некорректное размещение

        top, middle, bottom = placement['top'], placement['middle'], placement['bottom']

        # Используем обновленные функции scoring (они принимают int)
        if check_board_foul(top, middle, bottom):
            return -1, -1 # Фол

        stays_in_fl = check_fantasyland_stay(top, middle, bottom)
        total_royalty = (get_row_royalty(top, 'top') +
                         get_row_royalty(middle, 'middle') +
                         get_row_royalty(bottom, 'bottom'))

        score = 1 if stays_in_fl else 0
        return score, total_royalty

    def _find_best_hand(self, cards: List[int], n: int) -> Optional[List[int]]: # Типы карт - int
        """Находит лучшую n-карточную комбинацию из списка карт (int)."""
        if len(cards) < n: return None

        best_hand = None
        # Инициализируем худшим возможным рангом + запас
        best_rank = RANK_CLASS_HIGH_CARD + 1000
        found_hand = False

        # Перебираем комбинации
        for combo in combinations(cards, n):
            combo_list = list(combo)
            # Используем обновленную функцию scoring (она принимает int)
            rank = get_hand_rank_safe(combo_list)
            # Меньший ранг лучше
            if rank < best_rank:
                best_rank = rank
                best_hand = combo_list
                found_hand = True

        return list(best_hand) if found_hand else None

    def _try_build_strong_bottom(self, cards: List[int]) -> Optional[Dict[str, List[int]]]: # Типы карт - int
        """Пытается собрать Каре+ на боттоме, остальное эвристически."""
        if len(cards) != 13: return None

        best_stay_placement = None
        max_royalty = -1

        # Перебираем комбинации для боттома
        # Ограничиваем количество комбинаций для производительности
        bottom_combos = list(combinations(cards, 5))
        if len(bottom_combos) > 252: # 13 choose 5 = 1287, многовато. 252 = 10 choose 5
             bottom_combos = random.sample(bottom_combos, 252)

        for bottom_list in bottom_combos:
            # bottom_list = list(bottom_combo) # Уже список
            rank_b = get_hand_rank_safe(bottom_list)

            # Проверяем, является ли боттом Каре или лучше
            if rank_b <= RANK_CLASS_QUADS: # Каре = 166
                remaining8 = [c for c in cards if c not in bottom_list]
                if len(remaining8) != 8: continue

                # Находим лучшую среднюю руку из оставшихся 8
                middle_list = self._find_best_hand(remaining8, 5)
                if middle_list:
                    top_list = [c for c in remaining8 if c not in middle_list]
                    if len(top_list) == 3:
                        # Проверяем на фол
                        rank_m = get_hand_rank_safe(middle_list)
                        rank_t = get_hand_rank_safe(top_list)
                        if not (rank_b <= rank_m <= rank_t): continue # Фол

                        # Считаем роялти
                        royalty = (get_row_royalty(top_list, 'top') +
                                   get_row_royalty(middle_list, 'middle') +
                                   get_row_royalty(bottom_list, 'bottom'))

                        # Обновляем лучшее размещение с удержанием ФЛ
                        if royalty > max_royalty:
                            max_royalty = royalty
                            best_stay_placement = {'top': top_list, 'middle': middle_list, 'bottom': bottom_list}
        return best_stay_placement

    def _try_build_set_top(self, cards: List[int]) -> Optional[Dict[str, List[int]]]: # Типы карт - int
        """Пытается собрать Сет на топе, остальное эвристически."""
        if len(cards) != 13: return None

        best_stay_placement = None
        max_royalty = -1

        # Ищем возможные сеты
        # Используем RANK_MAP и CardUtils.get_rank_int из корневого card.py
        try:
             rank_counts = Counter(RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], -1) for c in cards)
             possible_set_ranks = [rank for rank, count in rank_counts.items() if count >= 3 and rank != -1]
        except (IndexError, KeyError, TypeError) as e:
             print(f"Warning: Error counting ranks for set top in FL solver: {e}. Hand: {cards}")
             possible_set_ranks = []


        for set_rank_index in possible_set_ranks:
            # Собираем карты для сета
            set_cards = [c for c in cards if RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], -1) == set_rank_index][:3]
            if len(set_cards) != 3: continue # На всякий случай

            remaining10 = [c for c in cards if c not in set_cards]
            if len(remaining10) != 10: continue

            # Находим лучшую нижнюю руку из оставшихся 10
            bottom_list = self._find_best_hand(remaining10, 5)
            if bottom_list:
                 middle_list = [c for c in remaining10 if c not in bottom_list]
                 if len(middle_list) == 5:
                     # Проверяем на фол
                     rank_b = get_hand_rank_safe(bottom_list)
                     rank_m = get_hand_rank_safe(middle_list)
                     rank_t = get_hand_rank_safe(set_cards)
                     if not (rank_b <= rank_m <= rank_t): continue # Фол

                     # Считаем роялти
                     royalty = (get_row_royalty(set_cards, 'top') +
                                get_row_royalty(middle_list, 'middle') +
                                get_row_royalty(bottom_list, 'bottom'))

                     # Обновляем лучшее размещение с удержанием ФЛ
                     if royalty > max_royalty:
                         max_royalty = royalty
                         best_stay_placement = {'top': set_cards, 'middle': middle_list, 'bottom': bottom_list}
        return best_stay_placement

    def _try_maximize_royalty_heuristic(self, cards: List[int]) -> Optional[Dict[str, List[int]]]: # Типы карт - int
        """Простая эвристика: размещаем лучшие возможные руки на боттом/мидл/топ без фола."""
        if len(cards) != 13: return None

        best_placement = None
        max_royalty = -1

        # Ограниченный перебор комбинаций для боттома
        bottom_combinations = list(combinations(cards, 5))
        if len(bottom_combinations) > 100: # Ограничение
            bottom_combinations = random.sample(bottom_combinations, 100)

        for bottom_list in bottom_combinations:
            # bottom_list = list(bottom_combo) # Уже список
            remaining8 = [c for c in cards if c not in bottom_list]
            if len(remaining8) != 8: continue

            # Находим лучшую среднюю руку
            middle_list = self._find_best_hand(remaining8, 5)
            if middle_list:
                 top_list = [c for c in remaining8 if c not in middle_list]
                 if len(top_list) == 3:
                     # Проверяем на фол
                     rank_b = get_hand_rank_safe(bottom_list)
                     rank_m = get_hand_rank_safe(middle_list)
                     rank_t = get_hand_rank_safe(top_list)
                     if not (rank_b <= rank_m <= rank_t): continue # Фол

                     # Считаем роялти
                     royalty = (get_row_royalty(top_list, 'top') +
                                get_row_royalty(middle_list, 'middle') +
                                get_row_royalty(bottom_list, 'bottom'))

                     # Обновляем лучшее размещение по роялти
                     if royalty > max_royalty:
                         max_royalty = royalty
                         best_placement = {'top': top_list, 'middle': middle_list, 'bottom': bottom_list}

        # Если перебор не дал результата, делаем простое размещение по силе
        if not best_placement:
             # Используем RANK_MAP и CardUtils.get_rank_int из корневого card.py
             try:
                  sorted_cards = sorted(cards, key=lambda c: RANK_MAP.get(STR_RANKS[CardUtils.get_rank_int(c)], 0), reverse=True)
                  placement = {'bottom': sorted_cards[0:5], 'middle': sorted_cards[5:10], 'top': sorted_cards[10:13]}
                  # Проверяем на фол
                  if not check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                      # Считаем роялти для этого простого размещения
                      royalty = (get_row_royalty(placement['top'], 'top') +
                                 get_row_royalty(placement['middle'], 'middle') +
                                 get_row_royalty(placement['bottom'], 'bottom'))
                      # Используем его, если оно лучше (хотя max_royalty должен быть -1)
                      if royalty > max_royalty:
                          best_placement = placement
             except (IndexError, KeyError, TypeError) as e:
                  print(f"Warning: Error creating simple fallback placement in FL solver: {e}")


        return best_placement
