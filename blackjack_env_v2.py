import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

def cmp(a, b):
    # Compara dos valores y devuelve 1 si a>b, -1 si a<b y 0 si son iguales
    return int((a > b)) - int((a < b))

# 1 = As, 2-10 num, J Q K valen 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(np_random):
    # Saca una carta del mazo al azar
    return np_random.choice(deck)

def draw_hand(np_random):
    # Saca dos cartas para empezar una mano
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):
    # Mira si hay un as que pueda valer 11 sin pasarse
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    # Suma real de la mano teniendo en cuenta el as usable
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    # True si la mano pasa de 21
    return sum_hand(hand) > 21

def score(hand):
    # 0 si bust, sino la suma real
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    # Devuelve True si las 2 primeras cartas suman 21 (Blackjack natural)
    return len(hand) == 2 and sorted(hand) in ([1,10],[10,1])

class BlackjackEnv(gym.Env):
    # Entorno de blackjack version 2 con premio por natural
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()
        self._reset()

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        # Inicializa la semilla del entorno
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # Lógica principal del turno del jugador
        assert self.action_space.contains(action), "Fallo action = {}".format(action)

        # Si el jugador pide carta
        if action:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                return self._get_obs(), -1, True, False, {}
            else:
                return self._get_obs(), 0, False, False, {}

        # Si el jugador se planta (action = 0)
        # Primero miramos si hay un natural
        player_nat = is_natural(self.player)
        dealer_nat = is_natural(self.dealer)

        # Caso especial natural
        if player_nat or dealer_nat:
            terminated = True
            if player_nat and not dealer_nat:
                reward = 1.5        # RECOMPENSA PEDIDA POR EL EJERCICIO 1.2
            elif dealer_nat and not player_nat:
                reward = -1
            else:
                reward = 0          # ambos natural
            return self._get_obs(), reward, terminated, False, {}

        # Si no hay naturals, juega el dealer normalmente
        while sum_hand(self.dealer) < 17:
            self.dealer.append(draw_card(self.np_random))

        reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, True, False, {}

    def _get_obs(self):
        # Estado que el agente ve
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        # Reparte las cartas y asegura minimo 12 para el jugador
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Logica original del entorno
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs(), {}

