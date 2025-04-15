# mcts_agent.py
import math
import time
import random
import multiprocessing
import traceback
from typing import Optional, Any, List, Tuple, Set
import sys # Добавлено для flush

from mcts_node import MCTSNode
from game_state import GameState
from fantasyland_solver import FantasylandSolver
from card import card_to_str, Card # Используем корневой card.py

# Функция-воркер для параллельного роллаута (без изменений)
def run_parallel_rollout(node_state_dict: dict) -> Tuple[float, Set[Any]]:
    try:
        game_state = GameState.from_dict(node_state_dict)
        if game_state.is_round_over():
             score_p0 = game_state.get_terminal_score()
             return float(score_p0), set()
        temp_node = MCTSNode(game_state)
        reward, sim_actions = temp_node.rollout(perspective_player=0)
        return reward, sim_actions
    except Exception as e:
        print(f"Error in parallel rollout worker: {e}")
        traceback.print_exc()
        return 0.0, set()


class MCTSAgent:
    """Агент MCTS для OFC Pineapple с RAVE и параллелизацией."""
    DEFAULT_EXPLORATION = 1.414
    DEFAULT_RAVE_K = 500
    DEFAULT_TIME_LIMIT_MS = 5000
    DEFAULT_NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    DEFAULT_ROLLOUTS_PER_LEAF = 4

    def __init__(self,
                 exploration: Optional[float] = None,
                 rave_k: Optional[float] = None,
                 time_limit_ms: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 rollouts_per_leaf: Optional[int] = None):

        self.exploration = exploration if exploration is not None else self.DEFAULT_EXPLORATION
        self.rave_k = rave_k if rave_k is not None else self.DEFAULT_RAVE_K
        time_limit_val = time_limit_ms if time_limit_ms is not None else self.DEFAULT_TIME_LIMIT_MS
        self.time_limit = time_limit_val / 1000.0
        max_cpus = multiprocessing.cpu_count()
        requested_workers = num_workers if num_workers is not None else self.DEFAULT_NUM_WORKERS
        self.num_workers = max(1, min(requested_workers, max_cpus))
        self.rollouts_per_leaf = rollouts_per_leaf if rollouts_per_leaf is not None else self.DEFAULT_ROLLOUTS_PER_LEAF
        if self.num_workers == 1 and self.rollouts_per_leaf > 1:
             print(f"Warning: num_workers=1, reducing rollouts_per_leaf from {self.rollouts_per_leaf} to 1.")
             self.rollouts_per_leaf = 1
        self.fantasyland_solver = FantasylandSolver()
        print(f"MCTS Agent initialized with: TimeLimit={self.time_limit:.2f}s, Exploration={self.exploration}, RaveK={self.rave_k}, Workers={self.num_workers}, RolloutsPerLeaf={self.rollouts_per_leaf}")
        try:
             current_method = multiprocessing.get_start_method(allow_none=True)
             if current_method != 'spawn':
                  multiprocessing.set_start_method('spawn', force=True)
        except Exception as e:
             print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default ({multiprocessing.get_start_method()}).")


    def choose_action(self, game_state: GameState) -> Optional[Any]:
        """Выбирает лучшее действие с помощью MCTS с параллелизацией."""
        print(f"--- AI choose_action called for state (Street {game_state.street}) ---")
        sys.stdout.flush()
        gs = game_state # Используем gs для краткости

        # Определяем игрока, для которого выбираем ход
        player_to_act = gs._get_player_to_move() # Используем метод из MCTSNode для консистентности

        if player_to_act == -1:
             print("Error: choose_action determined no player to act. Returning None.")
             sys.stdout.flush()
             return None

        print(f"Player to act: {player_to_act}")
        sys.stdout.flush()

        # --- Обработка хода в Fantasyland ---
        if gs.is_fantasyland_round and gs.fantasyland_status[player_to_act]:
             hand = gs.fantasyland_hands[player_to_act]
             if hand:
                 print(f"Player {player_to_act} solving Fantasyland...")
                 sys.stdout.flush()
                 start_fl_time = time.time()
                 placement, discarded = self.fantasyland_solver.solve(hand)
                 solve_time = time.time() - start_fl_time
                 print(f"Fantasyland solved in {solve_time:.3f}s")
                 sys.stdout.flush()
                 if placement:
                     return ("FANTASYLAND_PLACEMENT", placement, discarded)
                 else:
                     print("Warning: Fantasyland solver failed.")
                     sys.stdout.flush()
                     return ("FANTASYLAND_FOUL", hand)
             else:
                 print(f"Warning: choose_action called for FL player {player_to_act} but no hand found.")
                 sys.stdout.flush()
                 return None

        # --- Обычный ход MCTS ---
        print(f"Getting legal actions for player {player_to_act}...")
        sys.stdout.flush()
        initial_actions = gs.get_legal_actions_for_player(player_to_act)
        print(f"Found {len(initial_actions)} legal actions.")
        sys.stdout.flush()

        if not initial_actions:
            print(f"Warning: No legal actions found for player {player_to_act} in choose_action.")
            sys.stdout.flush()
            return None
        if len(initial_actions) == 1:
            print("Only one legal action, returning immediately.")
            sys.stdout.flush()
            return initial_actions[0]

        print("Initializing MCTS root node...")
        sys.stdout.flush()
        root_node = MCTSNode(gs)
        root_node.untried_actions = list(initial_actions)
        random.shuffle(root_node.untried_actions)
        for act in root_node.untried_actions:
             if act not in root_node.rave_visits:
                  root_node.rave_visits[act] = 0
                  root_node.rave_total_reward[act] = 0.0

        start_time = time.time()
        num_simulations = 0
        print("Starting MCTS simulations...")
        sys.stdout.flush()

        try:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                while time.time() - start_time < self.time_limit:
                    # --- Selection ---
                    # print("MCTS: Selection phase...") # Слишком много логов
                    path, leaf_node = self._select(root_node)
                    if leaf_node is None:
                         # print("MCTS: Selection failed.")
                         continue

                    # print(f"MCTS: Selected leaf node (Depth: {len(path)-1}, Terminal: {leaf_node.is_terminal()})")

                    results = []
                    simulation_actions_aggregated = set()
                    node_to_rollout_from = leaf_node
                    expanded_node = None

                    if not leaf_node.is_terminal():
                        # --- Expansion ---
                        if leaf_node.untried_actions:
                             # print("MCTS: Expansion phase...")
                             expanded_node = leaf_node.expand()
                             if expanded_node:
                                  # print("MCTS: Node expanded.")
                                  node_to_rollout_from = expanded_node
                                  path.append(expanded_node)
                             # else: print("MCTS: Expansion failed (no actions or error).")


                        # --- Parallel Rollouts ---
                        # print(f"MCTS: Rollout phase from node (Depth: {len(path)-1})...")
                        try:
                            node_state_dict = node_to_rollout_from.game_state.to_dict()
                        except Exception as e:
                             print(f"Error serializing state for parallel rollout: {e}")
                             sys.stdout.flush()
                             continue

                        async_results = [pool.apply_async(run_parallel_rollout, (node_state_dict,))
                                         for _ in range(self.rollouts_per_leaf)]

                        for res in async_results:
                            try:
                                timeout_get = max(0.1, self.time_limit * 0.1)
                                reward, sim_actions = res.get(timeout=timeout_get)
                                results.append(reward)
                                simulation_actions_aggregated.update(sim_actions)
                                num_simulations += 1
                            except multiprocessing.TimeoutError:
                                print("Warning: Rollout worker timed out.")
                                sys.stdout.flush()
                            except Exception as e:
                                print(f"Warning: Error getting result from worker: {e}")
                                sys.stdout.flush()

                    else: # Лист терминальный
                        # print("MCTS: Leaf node is terminal.")
                        reward = leaf_node.game_state.get_terminal_score()
                        results.append(reward)
                        num_simulations += 1

                    # --- Backpropagation ---
                    if results:
                        # print("MCTS: Backpropagation phase...")
                        total_reward_from_batch = sum(results)
                        num_rollouts_in_batch = len(results)
                        if expanded_node and expanded_node.action:
                             simulation_actions_aggregated.add(expanded_node.action)
                        self._backpropagate_parallel(path, total_reward_from_batch, num_rollouts_in_batch, simulation_actions_aggregated)

        except Exception as e:
             print(f"Error during MCTS parallel execution: {e}")
             traceback.print_exc()
             sys.stdout.flush()
             print("Choosing random action due to MCTS error.")
             return random.choice(initial_actions) if initial_actions else None
        finally:
             # Убедимся, что пул закрыт, даже если была ошибка
             # Контекстный менеджер 'with' должен это делать автоматически
             pass

        elapsed_time = time.time() - start_time
        print(f"MCTS finished: Ran {num_simulations} simulations in {elapsed_time:.3f}s ({num_simulations/elapsed_time:.1f} sims/s if elapsed_time > 0 else 0} sims/s) using {self.num_workers} workers.")
        sys.stdout.flush()

        # --- Выбор лучшего хода (ИСПРАВЛЕННЫЙ БЛОК) ---
        if not root_node.children:
            print("Warning: No children found in root node after MCTS, choosing random initial action.")
            sys.stdout.flush()
            return random.choice(initial_actions) if initial_actions else None

        best_action_robust = None
        max_visits = -1
        items = list(root_node.children.items())
        random.shuffle(items) # Случайность при равных посещениях

        print(f"Evaluating {len(items)} child nodes...")
        sys.stdout.flush()
        for action, child_node in items:
            # print(f"  Action: {self._format_action(action)}, Visits: {child_node.visits}") # Отладка
            if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_action_robust = action

        if best_action_robust is None:
             print("Warning: Could not determine best action based on visits (all 0?), choosing first shuffled.")
             sys.stdout.flush()
             # items уже перемешан, берем первое действие
             best_action_robust = items[0][0] if items else (random.choice(initial_actions) if initial_actions else None)


        if best_action_robust:
             print(f"Selected action based on max visits ({max_visits}): {self._format_action(best_action_robust)}")
        else:
             print("Error: Failed to select any action.")
        sys.stdout.flush()
        return best_action_robust
        # --- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА ---


    def _select(self, node: MCTSNode) -> Tuple[List[MCTSNode], Optional[MCTSNode]]:
        """Фаза выбора узла для расширения/симуляции."""
        path = [node]
        current_node = node
        while not current_node.is_terminal():
            player_to_move = current_node._get_player_to_move()
            if player_to_move == -1: return path, current_node # Терминальный или нет ходов

            if current_node.untried_actions is None:
                 current_node.untried_actions = current_node.game_state.get_legal_actions_for_player(player_to_move)
                 random.shuffle(current_node.untried_actions)
                 for act in current_node.untried_actions:
                     if act not in current_node.rave_visits:
                         current_node.rave_visits[act] = 0
                         current_node.rave_total_reward[act] = 0.0

            if current_node.untried_actions:
                return path, current_node # Узел для расширения

            if not current_node.children:
                 return path, current_node # Лист

            selected_child = current_node.uct_select_child(self.exploration, self.rave_k)
            if selected_child is None:
                # print(f"Warning: Selection returned None child from node {current_node}. Returning node as leaf.")
                if current_node.children:
                     try: selected_child = random.choice(list(current_node.children.values()))
                     except IndexError: return path, current_node
                else: return path, current_node
            current_node = selected_child
            path.append(current_node)
        return path, current_node


    def _backpropagate_parallel(self, path: List[MCTSNode], total_reward: float, num_rollouts: int, simulation_actions: Set[Any]):
        """Фаза обратного распространения для параллельных роллаутов."""
        if num_rollouts == 0: return

        for node in reversed(path):
            node.visits += num_rollouts
            node.total_reward += total_reward # Награда всегда с точки зрения P0

            player_to_move_from_node = node._get_player_to_move()
            if player_to_move_from_node != -1:
                 possible_actions_from_node = set(node.children.keys())
                 if node.untried_actions: possible_actions_from_node.update(node.untried_actions)
                 relevant_sim_actions = simulation_actions.intersection(possible_actions_from_node)

                 for action in relevant_sim_actions:
                      if action in node.rave_visits:
                           node.rave_visits[action] += num_rollouts
                           node.rave_total_reward[action] += total_reward
                      else:
                           node.rave_visits[action] = num_rollouts
                           node.rave_total_reward[action] = total_reward


    def _format_action(self, action: Any) -> str:
        """Форматирует действие для вывода."""
        # (Код форматирования без изменений)
        if action is None: return "None"
        try:
            if isinstance(action, tuple) and len(action) == 3 and \
               isinstance(action[0], tuple) and len(action[0]) == 3 and isinstance(action[0][0], int) and \
               isinstance(action[1], tuple) and len(action[1]) == 3 and isinstance(action[1][0], int) and \
               isinstance(action[2], int):
                p1, p2, d = action
                return f"PINEAPPLE: {card_to_str(p1[0])}@{p1[1]}{p1[2]}, {card_to_str(p2[0])}@{p2[1]}{p2[2]}; Discard {card_to_str(d)}"
            elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], list) and \
                 action[0] and isinstance(action[0][0], tuple) and len(action[0][0]) == 3 and isinstance(action[0][0][0], int):
                 placements_str = ", ".join([f"{card_to_str(c)}@{r}{i}" for c, r, i in action[0]])
                 return f"STREET 1: Place {placements_str}"
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT" and isinstance(action[1], list):
                 return f"FANTASYLAND_INPUT ({len(action[1])} cards)"
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT":
                 discard_count = len(action[2]) if isinstance(action[2], list) else '?'
                 return f"FANTASYLAND_PLACE (Discard {discard_count})"
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL":
                 discard_count = len(action[1]) if isinstance(action[1], list) else '?'
                 return f"FANTASYLAND_FOUL (Discard {discard_count})"
            else:
                 if isinstance(action, tuple):
                      formatted_items = []
                      for item in action:
                           if isinstance(item, (str, int, float, bool, type(None))): formatted_items.append(repr(item))
                           elif isinstance(item, Card): formatted_items.append(card_to_str(item))
                           elif isinstance(item, list): formatted_items.append("[...]")
                           elif isinstance(item, dict): formatted_items.append("{...}")
                           else: formatted_items.append(self._format_action(item))
                      return f"Unknown Tuple Action: ({', '.join(formatted_items)})"
                 return str(action)
        except Exception as e:
             return "ErrorFormattingAction"
