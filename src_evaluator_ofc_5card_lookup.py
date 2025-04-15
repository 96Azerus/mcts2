# -*- coding: utf-8 -*-
import itertools
# --- ИЗМЕНЕНИЕ: Импортируем корневой Card и его утилиты ---
try:
    from card import Card as CardUtils # Используем алиас
except ImportError:
    try:
         import sys
         import os
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
         if project_root not in sys.path:
              sys.path.insert(0, project_root)
         from card import Card as CardUtils
         print("DEBUG: Imported root card utils via sys.path modification in 5-card lookup.")
    except ImportError as e_inner:
         print(f"ОШИБКА: Не удалось импортировать корневой card.py в ofc_5card_lookup: {e_inner}. Используется запасной вариант.")
         class CardUtils: # Заглушка
             PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
             INT_RANKS = range(13)
             @staticmethod
             def prime_product_from_rankbits(rankbits):
                 p = 1
                 for i in CardUtils.INT_RANKS:
                     if rankbits & (1 << i): p *= CardUtils.PRIMES[i]
                 return p
             @staticmethod
             def prime_product_from_hand(card_ints):
                 p = 1
                 for c in card_ints: p *= (c & 0x3F) # Примерная логика get_prime
                 return p


class LookupTable(object):
    """
    Number of Distinct Hand Values:

    Straight Flush   10
    Four of a Kind   156      [(13 choose 2) * (2 choose 1)] -> 13 * 12 = 156
    Full Houses      156      [(13 choose 2) * (2 choose 1)] -> 13 * 12 = 156
    Flush            1277     [(13 choose 5) - 10 straight flushes] = 1287 - 10 = 1277
    Straight         10
    Three of a Kind  858      [(13 choose 3) * (3 choose 1)]? Нет, (13 choose 1) * (12 choose 2) = 13 * 66 = 858
    Two Pair         858      [(13 choose 2) * (11 choose 1)] = 78 * 11 = 858
    One Pair         2860     [(13 choose 1) * (12 choose 3)] = 13 * 220 = 2860
    High Card      + 1277     [(13 choose 5) - 10 straights] = 1287 - 10 = 1277
    -------------------------
    TOTAL            7462

    Here we create a lookup table which maps:
        5 card hand's unique prime product => rank in range [1, 7462]

    Examples:
    * Royal flush (best hand possible)          => 1
    * 7-5-4-3-2 unsuited (worst hand possible)  => 7462
    """
    MAX_STRAIGHT_FLUSH  = 10
    MAX_FOUR_OF_A_KIND  = 166 # 10 + 156
    MAX_FULL_HOUSE      = 322 # 166 + 156
    MAX_FLUSH           = 1599 # 322 + 1277
    MAX_STRAIGHT        = 1609 # 1599 + 10
    MAX_THREE_OF_A_KIND = 2467 # 1609 + 858
    MAX_TWO_PAIR        = 3325 # 2467 + 858
    MAX_PAIR            = 6185 # 3325 + 2860
    MAX_HIGH_CARD       = 7462 # 6185 + 1277

    MAX_TO_RANK_CLASS = {
        MAX_STRAIGHT_FLUSH: 1,
        MAX_FOUR_OF_A_KIND: 2,
        MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4,
        MAX_STRAIGHT: 5,
        MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7,
        MAX_PAIR: 8,
        MAX_HIGH_CARD: 9
    }

    RANK_CLASS_TO_STRING = {
        1 : "Straight Flush",
        2 : "Four of a Kind",
        3 : "Full House",
        4 : "Flush",
        5 : "Straight",
        6 : "Three of a Kind",
        7 : "Two Pair",
        8 : "Pair",
        9 : "High Card"
    }

    def __init__(self):
        """
        Calculates lookup tables using CardUtils from the root card.py
        """
        # create dictionaries
        self.flush_lookup = {}
        self.unsuited_lookup = {}

        # create the lookup table in piecewise fashion
        self.flushes()  # this will call straights and high cards method,
                        # we reuse some of the bit sequences
        self.multiples()

    def flushes(self):
        """
        Straight flushes and flushes.
        Lookup is done on 13 bit integer (2^13 > 7462):
        xxxbbbbb bbbbbbbb => integer hand index
        """

        # straight flushes in rank order (bitmask representation)
        straight_flushes = [
            7936, # 0b1111100000000, # A, K, Q, J, T
            3968, # 0b0111110000000, # K, Q, J, T, 9
            1984, # 0b0011111000000, # Q, J, T, 9, 8
            992,  # 0b0001111100000, # J, T, 9, 8, 7
            496,  # 0b0000111110000, # T, 9, 8, 7, 6
            248,  # 0b0000011111000, # 9, 8, 7, 6, 5
            124,  # 0b0000001111100, # 8, 7, 6, 5, 4
            62,   # 0b0000000111110, # 7, 6, 5, 4, 3
            31,   # 0b0000000011111, # 6, 5, 4, 3, 2
            4111  # 0b1000000001111, # A, 5, 4, 3, 2 (5 high)
        ]

        # Dynamically generate all other flushes (including straight flushes)
        flushes = []
        # Start from the lowest 5-bit pattern
        gen = self.get_lexographically_next_bit_sequence(int('0b11111', 2))

        # Total number of 5-card combinations from 13 ranks = 1287
        # We iterate 1287 times to get all combinations
        for _ in range(1287):
            try:
                 f = next(gen)
                 flushes.append(f)
            except StopIteration: # Should not happen if range is correct
                 break

        # Remove straight flushes from the flushes list
        flushes_without_straights = []
        straight_flush_set = set(straight_flushes)
        for f in flushes:
            if f not in straight_flush_set:
                flushes_without_straights.append(f)

        # Sort flushes in descending order (strongest first)
        # This sorting is implicit in the old implementation by reversing,
        # but explicit sorting by bit value is safer.
        flushes_without_straights.sort(reverse=True)

        # Add straight flushes to the lookup table (rank 1 to 10)
        rank = 1
        for sf in straight_flushes:
            # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
            prime_product = CardUtils.prime_product_from_rankbits(sf)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # Add normal flushes to the lookup table
        # Start ranking after full houses (rank 323)
        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f in flushes_without_straights:
            # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
            prime_product = CardUtils.prime_product_from_rankbits(f)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # Reuse these bit sequences for straights and high cards
        self.straight_and_highcards(straight_flushes, flushes_without_straights)

    def straight_and_highcards(self, straights, highcards):
        """
        Unique five card sets. Straights and highcards.
        Reuses bit sequences from flush calculations.
        """
        # Straights are ranked after flushes (rank 1600 to 1609)
        rank = LookupTable.MAX_FLUSH + 1
        for s in straights:
            # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
            prime_product = CardUtils.prime_product_from_rankbits(s)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

        # High cards are ranked after pairs (rank 6186 to 7462)
        rank = LookupTable.MAX_PAIR + 1
        # Sort highcards in descending order (strongest first)
        highcards.sort(reverse=True)
        for h in highcards:
            # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
            prime_product = CardUtils.prime_product_from_rankbits(h)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

    def multiples(self):
        """
        Pair, Two Pair, Three of a Kind, Full House, and 4 of a Kind.
        Uses prime products for lookup.
        """
        # --- ИЗМЕНЕНИЕ: Используем CardUtils из корневого card.py ---
        backwards_ranks = range(len(CardUtils.INT_RANKS) - 1, -1, -1)
        PRIMES = CardUtils.PRIMES

        # 1) Four of a Kind (ranks 11 to 166)
        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1
        for i in backwards_ranks: # Rank of the 4 cards
            # kicker rank
            kickers = list(backwards_ranks)
            kickers.remove(i)
            for k in kickers:
                product = PRIMES[i]**4 * PRIMES[k]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 2) Full House (ranks 167 to 322)
        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1
        for i in backwards_ranks: # Rank of the 3 cards
            # Rank of the pair
            pairranks = list(backwards_ranks)
            pairranks.remove(i)
            for pr in pairranks:
                product = PRIMES[i]**3 * PRIMES[pr]**2
                self.unsuited_lookup[product] = rank
                rank += 1

        # 3) Three of a Kind (ranks 1610 to 2467)
        rank = LookupTable.MAX_STRAIGHT + 1
        for r in backwards_ranks: # Rank of the 3 cards
            # Ranks of the two kickers
            kickers = list(backwards_ranks)
            kickers.remove(r)
            gen = itertools.combinations(kickers, 2)
            for k1, k2 in gen: # Kickers are sorted high to low
                product = PRIMES[r]**3 * PRIMES[k1] * PRIMES[k2]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 4) Two Pair (ranks 2468 to 3325)
        rank = LookupTable.MAX_THREE_OF_A_KIND + 1
        # Ranks of the two pairs
        tpgen = itertools.combinations(backwards_ranks, 2) # High pair, low pair
        for p1, p2 in tpgen:
            # Rank of the kicker
            kickers = list(backwards_ranks)
            kickers.remove(p1)
            kickers.remove(p2)
            for k in kickers:
                product = PRIMES[p1]**2 * PRIMES[p2]**2 * PRIMES[k]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 5) Pair (ranks 3326 to 6185)
        rank = LookupTable.MAX_TWO_PAIR + 1
        for pairrank in backwards_ranks: # Rank of the pair
            # Ranks of the three kickers
            kickers = list(backwards_ranks)
            kickers.remove(pairrank)
            kgen = itertools.combinations(kickers, 3) # Kickers sorted high to low
            for k1, k2, k3 in kgen:
                product = PRIMES[pairrank]**2 * PRIMES[k1] * PRIMES[k2] * PRIMES[k3]
                self.unsuited_lookup[product] = rank
                rank += 1

    def get_lexographically_next_bit_sequence(self, bits: int) -> int:
        """
        Bit hack from here:
        http://www-graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
        Generates the next lexicographical bit permutation.
        """
        t = (bits | (bits - 1)) + 1
        next_val = t | ((((t & -t) // (bits & -bits)) >> 1) - 1)
        yield next_val
        while True:
            t = (next_val | (next_val - 1)) + 1
            # Check if t is 0 to prevent division by zero if next_val becomes 0
            # Although in the context of 5 bits out of 13, next_val should not become 0
            if (next_val & -next_val) == 0: break # Safety break
            next_val = t | ((((t & -t) // (next_val & -next_val)) >> 1) - 1)
            # Check if we wrapped around (might happen if initial bits is invalid)
            if next_val <= bits: break
            yield next_val