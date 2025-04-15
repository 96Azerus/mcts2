# mcts_agent.py
import math
import time
import random
import multiprocessing # Добавляем импорт
import traceback # Для отладки ошибок
from typing import Optional, Any, List, Tuple, Set

from mcts_node import MCTSNode # Использует обновленный MCTSNode
from game_state import GameState # Использует обновленный GameState
from fantasyland_solver import FantasylandSolver # Использует обновленный FantasylandSolver
from card import card_to_str, Card # Используем корневой card.py

# Функция-воркер для параллельного роллаута (должна быть вне класса для pickle)
def run_parallel_rollout(node_state_dict: dict) -> Tuple[float, Set[Any]]:
    """Запускает один роллаут из переданного состояния узла."""
    # Восстанавливаем состояние и создаем временный узел
    try:
        # Используем GameState.from_dict для восстановления
        game_state = GameState.from_dict(node_state_dict)
        # Убедимся, что состояние не терминальное перед роллаутом
        if game_state.is_round_over():
             # Если терминальное, возвращаем счет напрямую
             score_p0 = game_state.get_terminal_score() # Использует обновленный scoring
             return float(score_p0), set()

        # Создаем временный узел для запуска роллаута
        temp_node = MCTSNode(game_state) # Parent и action не важны для роллаута
        # Запускаем роллаут с точки зрения игрока 0
        reward, sim_actions = temp_node.rollout(perspective_player=0) # Использует обновленный rollout
        return reward, sim_actions
    except Exception as e:
        print(f"Error in parallel rollout worker: {e}")
        traceback.print_exc()
        return 0.0, set() # Возвращаем нейтральный результат в случае ошибки


class MCTSAgent:
    """Агент MCTS для OFC Pineapple с RAVE и параллелизацией."""
    DEFAULT_EXPLORATION = 1.414
    DEFAULT_RAVE_K = 500
    DEFAULT_TIME_LIMIT_MS = 5000
    # Используем N-1 ядер, но не менее 1
    DEFAULT_NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    DEFAULT_ROLLOUTS_PER_LEAF = 4 # Количество роллаутов на лист за одну параллельную итерацию

    def __init__(self,
                 exploration: Optional[float] = None,
                 rave_k: Optional[float] = None,
                 time_limit_ms: Optional[int] = None,
                 num_workers: Optional[int] = None, # Параметр для кол-ва воркеров
                 rollouts_per_leaf: Optional[int] = None): # Параметр для кол-ва роллаутов на лист

        self.exploration = exploration if exploration is not None else self.DEFAULT_EXPLORATION
        self.rave_k = rave_k if rave_k is not None else self.DEFAULT_RAVE_K
        time_limit_val = time_limit_ms if time_limit_ms is not None else self.DEFAULT_TIME_LIMIT_MS
        self.time_limit = time_limit_val / 1000.0 # Конвертируем в секунды
        # Ограничиваем num_workers максимальным количеством ядер
        max_cpus = multiprocessing.cpu_count()
        requested_workers = num_workers if num_workers is not None else self.DEFAULT_NUM_WORKERS
        self.num_workers = max(1, min(requested_workers, max_cpus))

        self.rollouts_per_leaf = rollouts_per_leaf if rollouts_per_leaf is not None else self.DEFAULT_ROLLOUTS_PER_LEAF
        # Уменьшаем rollouts_per_leaf, если воркеров мало, чтобы избежать простоя
        if self.num_workers == 1 and self.rollouts_per_leaf > 1:
             print(f"Warning: num_workers=1, reducing rollouts_per_leaf from {self.rollouts_per_leaf} to 1.")
             self.rollouts_per_leaf = 1

        self.fantasyland_solver = FantasylandSolver() # Использует обновленный solver
        print(f"MCTS Agent initialized with: TimeLimit={self.time_limit:.2f}s, Exploration={self.exploration}, RaveK={self.rave_k}, Workers={self.num_workers}, RolloutsPerLeaf={self.rollouts_per_leaf}")

        # Устанавливаем метод старта процессов (важно для некоторых ОС и окружений)
        try:
             current_method = multiprocessing.get_start_method(allow_none=True)
             if current_method != 'spawn':
                  # print(f"Attempting to set multiprocessing start method to 'spawn' (current: {current_method}).")
                  multiprocessing.set_start_method('spawn', force=True)
                  # print(f"Multiprocessing start method set to: {multiprocessing.get_start_method()}")
        except Exception as e:
             print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default ({multiprocessing.get_start_method()}).")


    def choose_action(self, game_state: GameState) -> Optional[Any]:
        """Выбирает лучшее действие с помощью MCTS с параллелизацией."""
        # Определяем игрока, для которого выбираем ход
        player_to_act = -1
        gs = game_state
        if gs.is_round_over(): return None # Нет ходов в конце раунда

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

        # Если не удалось определить игрока (например, все ждут карт)
        if player_to_act == -1:
             print("Warning: choose_action could not determine player to act (likely waiting for deal). Returning None.")
             return None

        # --- Обработка хода в Fantasyland ---
        if game_state.is_fantasyland_round and game_state.fantasyland_status[player_to_act]:
             hand = game_state.fantasyland_hands[player_to_act] # Рука - список int
             if hand:
                 # print(f"Player {player_to_act} solving Fantasyland...")
                 start_fl_time = time.time()
                 # Вызываем обновленный солвер
                 placement, discarded = self.fantasyland_solver.solve(hand)
                 solve_time = time.time() - start_fl_time
                 # print(f"Fantasyland solved in {solve_time:.3f}s")
                 if placement:
                     # Возвращаем действие в формате для apply_fantasyland_placement
                     return ("FANTASYLAND_PLACEMENT", placement, discarded)
                 else:
                     print("Warning: Fantasyland solver failed.")
                     # Возвращаем действие для фола
                     return ("FANTASYLAND_FOUL", hand)
             else: # Руки нет (уже сыграна?)
                 print(f"Warning: choose_action called for FL player {player_to_act} but no hand found.")
                 return None

        # --- Обычный ход MCTS ---
        # Получаем легальные действия для игрока
        initial_actions = game_state.get_legal_actions_for_player(player_to_act)
        if not initial_actions:
             print(f"Warning: No legal actions found for player {player_to_act} in choose_action.")
             return None # Нет легальных ходов
        if len(initial_actions) == 1:
             # print("Only one legal action, returning immediately.")
             return initial_actions[0] # Единственный ход

        # Создаем корневой узел MCTS
        root_node = MCTSNode(game_state) # Использует обновленный MCTSNode
        # Инициализируем неиспробованные действия и RAVE
        root_node.untried_actions = list(initial_actions)
        random.shuffle(root_node.untried_actions)
        for act in root_node.untried_actions:
             if act not in root_node.rave_visits:
                  root_node.rave_visits[act] = 0
                  root_node.rave_total_reward[act] = 0.0

        start_time = time.time()
        num_simulations = 0

        try:
            # Используем контекстный менеджер для пула процессов
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # Основной цикл MCTS (пока не истечет время)
                while time.time() - start_time < self.time_limit:
                    # --- Selection ---
                    # Выбираем путь от корня до листа по UCT/RAVE
                    path, leaf_node = self._select(root_node)
                    if leaf_node is None: continue # Ошибка выбора

                    results = [] # Результаты роллаутов из этого листа
                    simulation_actions_aggregated = set() # Действия из всех роллаутов
                    node_to_rollout_from = leaf_node
                    expanded_node = None

                    # Если лист не терминальный
                    if not leaf_node.is_terminal():
                        # --- Expansion (попытка) ---
                        # Если есть неиспробованные действия, расширяем узел
                        if leaf_node.untried_actions:
                             expanded_node = leaf_node.expand() # Использует обновленный expand
                             if expanded_node:
                                  node_to_rollout_from = expanded_node # Роллаут из нового узла
                                  path.append(expanded_node) # Добавляем новый узел в путь

                        # --- Parallel Rollouts ---
                        try:
                            # Сериализуем состояние для передачи воркерам
                            node_state_dict = node_to_rollout_from.game_state.to_dict()
                        except Exception as e:
                             print(f"Error serializing state for parallel rollout: {e}")
                             continue # Пропускаем итерацию

                        # Запускаем несколько роллаутов параллельно
                        async_results = [pool.apply_async(run_parallel_rollout, (node_state_dict,))
                                         for _ in range(self.rollouts_per_leaf)]

                        # Собираем результаты роллаутов
                        for res in async_results:
                            try:
                                # Ожидаем результат с таймаутом
                                timeout_get = max(0.1, self.time_limit * 0.1)
                                reward, sim_actions = res.get(timeout=timeout_get)
                                results.append(reward)
                                simulation_actions_aggregated.update(sim_actions)
                                num_simulations += 1
                            except multiprocessing.TimeoutError:
                                print("Warning: Rollout worker timed out.")
                            except Exception as e:
                                print(f"Warning: Error getting result from worker: {e}")

                    else: # Лист терминальный
                        # Получаем счет напрямую из терминального состояния
                        reward = leaf_node.game_state.get_terminal_score() # Использует обновленный scoring
                        results.append(reward)
                        num_simulations += 1

                    # --- Backpropagation ---
                    # Обновляем статистику узлов вдоль пути
                    if results:
                        total_reward_from_batch = sum(results)
                        num_rollouts_in_batch = len(results)
                        # Добавляем действие, которое привело к расширенному узлу, в набор для RAVE
                        if expanded_node and expanded_node.action:
                             simulation_actions_aggregated.add(expanded_node.action)
                        # Вызываем обновленный backpropagate
                        self._backpropagate_parallel(path, total_reward_from_batch, num_rollouts_in_batch, simulation_actions_aggregated)

        except Exception as e:
             print(f"Error during MCTS parallel execution: {e}")
             traceback.print_exc()
             # В случае ошибки возвращаем случайный ход
             return random.choice(initial_actions) if initial_actions else None

        elapsed_time = time.time() - start_time
        # print(f"MCTS ran {num_simulations} simulations in {elapsed_time:.3f}s ({num_simulations/elapsed_time:.1f} sims/s) using {self.num_workers} workers.")

        # --- Выбор лучшего хода ---
        if not root_node.children:
            # Если нет дочерних узлов (например, время вышло до первой симуляции)
            return random.choice(initial_actions) if initial_actions else None

        # Выбираем самый посещаемый узел (более робастный выбор)
        best_action_robust = max(root_node.children, key=lambda act: root_node.children[act].visits)
        # Можно также выбрать по максимальному Q-value:
        # best_action_greedy = max(root_node.children, key=lambda act: root_node.children[act].get_q_value(player_to_act))

        return best_action_robust


    def _select(self, node: MCTSNode) -> Tuple[List[MCTSNode], Optional[MCTSNode]]:
        """Фаза выбора узла для расширения/симуляции."""
        path = [node]
        current_node = node
        while not current_node.is_terminal():
            # Определяем, кто ходит из текущего узла
            player_to_move = current_node._get_player_to_move() # Использует обновленный _get_player_to_move
            if player_to_move == -1: return path, current_node # Достигли терминального узла

            # Инициализируем действия, если еще не сделано
            if current_node.untried_actions is None:
                 current_node.untried_actions = current_node.game_state.get_legal_actions_for_player(player_to_move)
                 random.shuffle(current_node.untried_actions)
                 # Инициализация RAVE для новых действий
                 for act in current_node.untried_actions:
                     if act not in current_node.rave_visits:
                         current_node.rave_visits[act] = 0
                         current_node.rave_total_reward[act] = 0.0

            # Если есть неиспробованные действия, выбираем этот узел для расширения
            if current_node.untried_actions:
                return path, current_node

            # Если нет неиспробованных действий и нет дочерних узлов - это лист
            if not current_node.children:
                 return path, current_node

            # Если есть дочерние узлы, выбираем лучший по UCT/RAVE
            selected_child = current_node.uct_select_child(self.exploration, self.rave_k) # Использует обновленный uct_select_child
            if selected_child is None:
                # print(f"Warning: Selection returned None child from node {current_node}. Returning node as leaf.")
                # Пытаемся выбрать случайного потомка, если выбор не удался
                if current_node.children:
                     try: selected_child = random.choice(list(current_node.children.values()))
                     except IndexError: return path, current_node # Не должно произойти, если children не пуст
                else: return path, current_node # Если потомков нет (хотя проверка выше)
            current_node = selected_child
            path.append(current_node)

        # Дошли до терминального узла по ходу выбора
        return path, current_node


    def _backpropagate_parallel(self, path: List[MCTSNode], total_reward: float, num_rollouts: int, simulation_actions: Set[Any]):
        """Фаза обратного распространения для параллельных роллаутов."""
        if num_rollouts == 0: return

        # Проходим по пути от листа к корню
        for node in reversed(path):
            node.visits += num_rollouts # Увеличиваем посещения на количество роллаутов
            # Награда обновляется с точки зрения игрока 0 (как она пришла из роллаута)
            node.total_reward += total_reward

            # Обновляем RAVE статистику для действий, которые были сделаны в симуляциях
            # Определяем, чей ход из этого узла
            player_to_move_from_node = node._get_player_to_move()
            if player_to_move_from_node != -1: # Не обновляем RAVE для терминального узла
                 # Находим все возможные действия из этого узла (испробованные и нет)
                 possible_actions_from_node = set(node.children.keys())
                 if node.untried_actions: possible_actions_from_node.update(node.untried_actions)

                 # Находим пересечение возможных действий и действий из симуляций
                 relevant_sim_actions = simulation_actions.intersection(possible_actions_from_node)

                 # Обновляем RAVE для релевантных действий
                 for action in relevant_sim_actions:
                      if action in node.rave_visits:
                           node.rave_visits[action] += num_rollouts
                           # RAVE награда хранится с точки зрения игрока 0
                           node.rave_total_reward[action] += total_reward
                      else: # Инициализируем, если действие новое (маловероятно здесь)
                           node.rave_visits[action] = num_rollouts
                           node.rave_total_reward[action] = total_reward


    def _format_action(self, action: Any) -> str:
        """Форматирует действие для вывода."""
        if action is None: return "None"
        try:
            # Pineapple action: ((card1, r1, i1), (card2, r2, i2), discard)
            if isinstance(action, tuple) and len(action) == 3 and \
               isinstance(action[0], tuple) and len(action[0]) == 3 and isinstance(action[0][0], int) and \
               isinstance(action[1], tuple) and len(action[1]) == 3 and isinstance(action[1][0], int) and \
               isinstance(action[2], int):
                p1, p2, d = action
                return f"PINEAPPLE: {card_to_str(p1[0])}@{p1[1]}{p1[2]}, {card_to_str(p2[0])}@{p2[1]}{p2[2]}; Discard {card_to_str(d)}"
            # Street 1 action: ([(card, r, i), ...], [])
            elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], list) and \
                 action[0] and isinstance(action[0][0], tuple) and len(action[0][0]) == 3 and isinstance(action[0][0][0], int):
                 placements_str = ", ".join([f"{card_to_str(c)}@{r}{i}" for c, r, i in action[0]])
                 return f"STREET 1: Place {placements_str}"
            # Fantasyland input for AI: ("FANTASYLAND_INPUT", [card1, card2, ...])
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT" and isinstance(action[1], list):
                 return f"FANTASYLAND_INPUT ({len(action[1])} cards)"
            # Fantasyland placement result: ("FANTASYLAND_PLACEMENT", {row: [cards...]}, [discarded...])
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT":
                 discard_count = len(action[2]) if isinstance(action[2], list) else '?'
                 return f"FANTASYLAND_PLACE (Discard {discard_count})"
            # Fantasyland foul result: ("FANTASYLAND_FOUL", [hand...])
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL":
                 discard_count = len(action[1]) if isinstance(action[1], list) else '?'
                 return f"FANTASYLAND_FOUL (Discard {discard_count})"
            else:
                 # Общий случай для неизвестных кортежей
                 if isinstance(action, tuple):
                      formatted_items = []
                      for item in action:
                           if isinstance(item, (str, int, float, bool, type(None))): formatted_items.append(repr(item))
                           elif isinstance(item, Card): formatted_items.append(card_to_str(item)) # Маловероятно, т.к. карты int
                           elif isinstance(item, list): formatted_items.append("[...]")
                           elif isinstance(item, dict): formatted_items.append("{...}")
                           else: formatted_items.append(self._format_action(item)) # Рекурсивно
                      return f"Unknown Tuple Action: ({', '.join(formatted_items)})"
                 # Если не кортеж, просто строка
                 return str(action)
        except Exception as e:
             # print(f"Error formatting action {action}: {e}")
             return "ErrorFormattingAction"