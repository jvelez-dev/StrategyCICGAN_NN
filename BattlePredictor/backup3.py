from absl import app, flags
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
from terran_agent import TerranAgent
import random
import csv
import os
import sys
import multiprocessing as mp

# Configuración del entorno para desactivar la renderización
os.environ["SC2_RENDERER"] = "0"
os.environ["SC2_NO_DISPLAY"] = "1"
os.environ["PYSC2_NO_RENDER"] = "1"

# Parámetro para la cantidad de registros deseados
TOTAL_REGISTROS_DESEADOS = 100
NÚCLEOS_CPU = mp.cpu_count()  # Número de procesos en paralelo

# Clase para gestionar el estado de la batalla
class BattleManager:
    predictor_marines = 0
    enemy_marines = 0
    predictor_ready = False
    enemy_ready = False
    predictor_minerals = 0
    predictor_gas = 0
    predictor_workers = 0
    predictor_military_units = 0
    predictor_buildings = 0
    predictor_attack_time = 0
    predictor_strategy = ""
    predictor_damage_inflicted = 0
    predictor_opponent_resources = 0
    predictor_result = 0

    # Atributos del enemigo
    enemy_minerals = 0
    enemy_gas = 0
    enemy_workers = 0
    enemy_military_units = 0
    enemy_buildings = 0
    enemy_attack_time = 0
    enemy_strategy = ""
    enemy_damage_inflicted = 0
    enemy_opponent_resources = 0
    enemy_result = 0

# Agente enemigo
class EnemyAgent(TerranAgent):
    def __init__(self, bm):
        super(EnemyAgent, self).__init__()
        self.bm = bm

    def step(self, obs):
        super(EnemyAgent, self).step(obs)

        if obs.first():
            self.bm.enemy_ready = False
            self.bm.enemy_marines = random.randint(1, 10)
            self.bm.enemy_minerals = obs.observation.player.minerals
            self.bm.enemy_gas = obs.observation.player.vespene
            self.bm.enemy_workers = obs.observation.player.food_workers
            self.bm.enemy_military_units = len(self.marines)
            self.bm.enemy_buildings = len(self.buildings)
            self.bm.enemy_attack_time = 0
            self.bm.enemy_strategy = "attack" if self.bm.enemy_marines > 5 else "defense"
            self.bm.enemy_damage_inflicted = 0
            self.bm.enemy_opponent_resources = 0
            self.bm.enemy_result = 0

        if obs.last():
            self.bm.enemy_result = 1 if obs.reward > 0 else 0
            with open("tvt.csv", "a", newline="\n") as myfile:
                csvwriter = csv.writer(myfile)
                csvwriter.writerow([
                    self.bm.enemy_marines,
                    self.bm.predictor_marines,
                    obs.reward,
                    self.bm.enemy_minerals,
                    self.bm.enemy_gas,
                    self.bm.enemy_workers,
                    self.bm.enemy_military_units,
                    self.bm.enemy_buildings,
                    self.bm.enemy_attack_time,
                    self.bm.enemy_strategy,
                    self.bm.enemy_damage_inflicted,
                    self.bm.enemy_opponent_resources,
                    self.bm.enemy_result
                ])

        if len(self.marines) == self.bm.enemy_marines and not self.bm.enemy_ready:
            self.bm.enemy_ready = True
            self.bm.enemy_attack_time = obs.observation.game_loop

        if self.bm.predictor_ready and self.bm.enemy_ready:
            return self.attack()
        if self.supply_depot is None:
            return self.build_supply_depot()
        if self.barracks is None:
            return self.build_barracks()
        if len(self.marines) + self.queued_marine_count < self.bm.enemy_marines:
            return self.train_marine()
        return actions.RAW_FUNCTIONS.no_op()

# Agente predictor
class PredictorAgent(TerranAgent):
    def __init__(self, bm):
        super(PredictorAgent, self).__init__()
        self.bm = bm

    def step(self, obs):
        super(PredictorAgent, self).step(obs)

        if obs.first():
            self.bm.predictor_ready = False
            self.bm.predictor_marines = random.randint(1, 10)
            self.bm.predictor_minerals = obs.observation.player.minerals
            self.bm.predictor_gas = obs.observation.player.vespene
            self.bm.predictor_workers = obs.observation.player.food_workers
            self.bm.predictor_military_units = len(self.marines)
            self.bm.predictor_buildings = len(self.buildings)
            self.bm.predictor_attack_time = 0
            self.bm.predictor_strategy = "attack" if self.bm.predictor_marines > 5 else "defense"
            self.bm.predictor_damage_inflicted = 0
            self.bm.predictor_opponent_resources = 0
            self.bm.predictor_result = 0

        if len(self.marines) == self.bm.predictor_marines and not self.bm.predictor_ready:
            self.bm.predictor_ready = True
            self.bm.predictor_attack_time = obs.observation.game_loop

        if self.bm.predictor_ready and self.bm.enemy_ready:
            return self.attack()
        if self.supply_depot is None:
            return self.build_supply_depot()
        if self.barracks is None:
            return self.build_barracks()
        if len(self.marines) + self.queued_marine_count < self.bm.predictor_marines:
            return self.train_marine()
        return actions.RAW_FUNCTIONS.no_op()

# Función para contar los registros en el CSV
def contar_registros():
    if not os.path.exists("tvt.csv"):
        return 0
    with open("tvt.csv", "r") as file:
        return sum(1 for _ in file) - 1  # Resta 1 para ignorar la cabecera

# Función para ejecutar episodios
def run_episode(_):
    flags.FLAGS([sys.argv[0]])  # Inicializar los flags correctamente
    bm = BattleManager()
    agent1 = PredictorAgent(bm)
    agent2 = EnemyAgent(bm)

    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64,
            ),
            step_mul=1024,
            disable_fog=True,
            realtime=False,
            visualize=False,
            ensure_available_actions=False,
            game_steps_per_episode=30000
        ) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=1)
    except KeyboardInterrupt:
        pass

# Función principal
def main(unused_argv):
    # Si el archivo no existe, escribir encabezado
    if not os.path.exists("tvt.csv"):
        with open("tvt.csv", "w", newline="\n") as myfile:
            csvwriter = csv.writer(myfile)
            csvwriter.writerow([
                "enemy_marines",
                "predictor_marines",
                "reward",
                "enemy_minerals",
                "enemy_gas",
                "enemy_workers",
                "enemy_military_units",
                "enemy_buildings",
                "enemy_attack_time",
                "enemy_strategy",
                "enemy_damage_inflicted",
                "enemy_opponent_resources",
                "enemy_result"
            ])

    # Bucle hasta alcanzar los registros deseados
    while contar_registros() < TOTAL_REGISTROS_DESEADOS:
        num_episodios = min(5, TOTAL_REGISTROS_DESEADOS - contar_registros())  # Ejecutar lotes de 5
        with mp.Pool(processes=NÚCLEOS_CPU) as pool:
            pool.map(run_episode, range(num_episodios))

if __name__ == "__main__":
    mp.set_start_method("spawn")  # Compatibilidad con Windows
    main([])