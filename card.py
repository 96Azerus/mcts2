# card.py (Корневой)
# ИСПРАВЛЕНО: RANK_MAP вынесен на уровень модуля

from typing import List, Optional # Импортируем нужные типы

# --- Константы на уровне модуля ---
STR_RANKS = '23456789TJQKA'
INT_RANKS = range(13)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
RANK_CHAR_TO_INT = dict(zip(list(STR_RANKS), INT_RANKS))
SUIT_CHAR_TO_INT = {
    's' : 1, # spades
    'h' : 2, # hearts
    'd' : 4, # diamonds
    'c' : 8, # clubs
}
# --- RANK_MAP теперь на уровне модуля ---
RANK_MAP = RANK_CHAR_TO_INT
INT_SUIT_TO_CHAR = 'xshxdxxxc' # index 1='s', 2='h', 4='d', 8='c'
PRETTY_SUITS = {
    1: u"\u2660",  # spades
    2: u"\u2764",  # hearts
    4: u"\u2666",  # diamonds
    8: u"\u2663"  # clubs
}
PRETTY_REDS = [2, 4] # hearts and diamonds


class Card: # Оставляем класс для удобства использования статических методов
    """
    Static class that handles cards. We represent cards as 32-bit integers...
    (Описание класса без изменений)
    """

    # Используем константы, определенные ВЫШЕ на уровне модуля
    STR_RANKS = STR_RANKS
    INT_RANKS = INT_RANKS
    PRIMES = PRIMES
    RANK_CHAR_TO_INT = RANK_CHAR_TO_INT
    SUIT_CHAR_TO_INT = SUIT_CHAR_TO_INT
    RANK_MAP = RANK_MAP # Можно оставить и здесь для доступа через Card.RANK_MAP, если нужно
    INT_SUIT_TO_CHAR = INT_SUIT_TO_CHAR
    PRETTY_SUITS = PRETTY_SUITS
    PRETTY_REDS = PRETTY_REDS

    @staticmethod
    def new(string: str) -> int:
        """
        Converts Card string to binary integer representation of card.
        """
        if len(string) != 2:
            raise ValueError(f"Invalid card string: '{string}'. Expected 2 characters (e.g., 'As', 'Td').")

        rank_char = string[0].upper()
        suit_char = string[1].lower()

        # Используем константы модуля напрямую или через self/Card
        if rank_char not in Card.RANK_CHAR_TO_INT:
             raise ValueError(f"Invalid rank character: '{rank_char}' in card string '{string}'.")
        if suit_char not in Card.SUIT_CHAR_TO_INT:
             raise ValueError(f"Invalid suit character: '{suit_char}' in card string '{string}'.")

        rank_int = Card.RANK_CHAR_TO_INT[rank_char]
        suit_int = Card.SUIT_CHAR_TO_INT[suit_char]
        rank_prime = Card.PRIMES[rank_int]

        bitrank = 1 << rank_int << 16
        suit = suit_int << 12
        rank = rank_int << 8

        return bitrank | suit | rank | rank_prime

    @staticmethod
    def int_to_str(card_int: int) -> str:
        """Converts integer card representation to string."""
        if not isinstance(card_int, int) or card_int < 0:
             return "__"
        rank_int = Card.get_rank_int(card_int)
        suit_int = Card.get_suit_int(card_int)
        if not (0 <= rank_int < len(Card.STR_RANKS)):
             return "__"
        if not (0 <= suit_int < len(Card.INT_SUIT_TO_CHAR)) or Card.INT_SUIT_TO_CHAR[suit_int] == 'x':
             return "__"
        return Card.STR_RANKS[rank_int] + Card.INT_SUIT_TO_CHAR[suit_int]

    @staticmethod
    def get_rank_int(card_int: int) -> int:
        """Extracts rank index (0-12) from integer card."""
        return (card_int >> 8) & 0xF

    @staticmethod
    def get_suit_int(card_int: int) -> int:
        """Extracts suit integer (1, 2, 4, 8) from integer card."""
        return (card_int >> 12) & 0xF

    @staticmethod
    def get_bitrank_int(card_int: int) -> int:
        """Extracts rank bitmask from integer card."""
        return (card_int >> 16) & 0x1FFF

    @staticmethod
    def get_prime(card_int: int) -> int:
        """Extracts rank prime from integer card."""
        return card_int & 0x3F

    @staticmethod
    def prime_product_from_hand(card_ints: List[int]) -> int:
        """ Calculates the prime product from a list of integer cards. """
        product = 1
        for c in card_ints:
            product *= Card.get_prime(c)
        return product

    @staticmethod
    def prime_product_from_rankbits(rankbits: int) -> int:
        """ Calculates the prime product from the rank bits. """
        product = 1
        for i in Card.INT_RANKS:
            if rankbits & (1 << i):
                product *= Card.PRIMES[i]
        return product

    @staticmethod
    def hand_to_binary(card_strs: List[str]) -> List[int]:
        """ Converts a list of card strings to a list of integer representations. """
        return [Card.new(c) for c in card_strs]

    @staticmethod
    def int_to_pretty_str(card_int: int) -> str:
        """ Creates a human-readable string with suit symbols. """
        rank_int = Card.get_rank_int(card_int)
        suit_int = Card.get_suit_int(card_int)

        if not (0 <= rank_int < len(Card.STR_RANKS)): return "[?]"
        suit_sym = Card.PRETTY_SUITS.get(suit_int)
        if suit_sym is None: return "[?]"

        r = Card.STR_RANKS[rank_int]
        s = suit_sym

        try:
            from termcolor import colored
            if suit_int in Card.PRETTY_REDS:
                s = colored(s, "red")
        except ImportError:
            pass

        return f"[{r}{s}]"

    @staticmethod
    def print_pretty_card(card_int: int):
        """ Prints a single pretty card string to the console. """
        print(Card.int_to_pretty_str(card_int))

    @staticmethod
    def print_pretty_cards(card_ints: List[int]):
        """ Prints a list of pretty card strings to the console. """
        print(" ".join(Card.int_to_pretty_str(c) for c in card_ints))

# --- Функции-алиасы на уровне модуля ---
def card_from_str(string: str) -> int:
    """Alias for Card.new()"""
    return Card.new(string)

def card_to_str(card_int: Optional[int]) -> str:
    """Alias for Card.int_to_str(), handles None."""
    if card_int is None:
        return "__"
    return Card.int_to_str(card_int)
