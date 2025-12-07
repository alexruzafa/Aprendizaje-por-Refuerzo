import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

def cmp(a, b):
    """
    Compara dos numeros y devuelve 1 si a>b, -1 si a<b y 0 si son iguales.
    """
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    """
    Saca una carta aleatoria del mazo usando el generador np_random.
    """
    return np_random.choice(deck)


def draw_hand(np_random):
    """
    Genera una mano inicial de blackjack con dos cartas.
    """
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):
    """
    Comprueba si en la mano hay un as que se puede contar como 11 sin pasarse de 21.
    """   
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    """
    Calcula la suma real de la mano teniendo en cuenta si hay as usable (vale 11).
    """
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    """
    Indica si la mano se pasa de 21 puntos (el jugador hace bust).
    """
    return sum_hand(hand) > 21


def score(hand): 
    """
    Devuelve la puntuacion final de una mano, 0 si esta en bust o la suma de la mano en caso contrario.
    """
    return 0 if is_bust(hand) else sum_hand(hand)


class BlackjackEnv(gym.Env):
    """
    Entorno de blackjack para RL: define espacios de acciones y estados y la logica basica del juego.
    """
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
        """
        Inicializa la semilla del generador aleatorio del entorno.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        Ejecuta una accion del agente (pedir carta o plantarse) y devuelve
        la nueva observacion, la recompensa, si el episodio termina y la info extra.
        """
        assert self.action_space.contains(action), "Fallo, Action = {}".format(action)
        if action:  # el jugador pide carta (HIT)
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1
            else:
                terminated = False
                reward = 0
        else:  # el jugador se planta (STICK) y juega la banca
            terminated = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """
        Construye el estado que ve el agente: suma del jugador, carta visible del dealer y si hay as usable.
        """
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        """
        Reinicia la partida: reparte las cartas iniciales a jugador y dealer y
        asegura que la suma del jugador sea al menos 12. Devuelve el estado inicial.
        """
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs(), {}
