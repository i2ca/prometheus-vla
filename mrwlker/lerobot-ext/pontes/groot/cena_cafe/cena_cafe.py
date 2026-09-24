"""A cena do café no simulador de G1 da NVIDIA (robocasa + controle de corpo inteiro).

Importar este módulo registra, no robocasa e no gym, a cena

    gr00tlocomanip_g1_sim/LMCafeXicaraCoador_G1_gear_wbc

(e as variantes `G1ArmsOnly`, `G1FixedLowerBody`, `G1FixedBase`, como as deles).
Duas mesas, o robô em pé na frente da principal, e os nossos objetos no lugar dos
deles. Os objetos saem de `gera_objetos.py`.

    MESA DO COADOR (y=1,5)          MESA PRINCIPAL (tampo 1,58 x 0,74 m, centro x=0,5)
    ┌──────────────┐    ┌─────────────────────────────────────────────┐  +x
    │    coador    │    │  chaleira                        pote+tampa │
    │              │    │                  scoop            xícara    │
    └──────────────┘    └─────────────────────────────────────────────┘
            +y (esquerda)                robô                -y (direita)

É o MESMO arranjo da `LMPnPAppleToPlate` deles — pegar na mesa da frente, andar
para a esquerda e pôr na outra mesa — com a xícara no lugar da maçã e o coador
no lugar do prato. De propósito: é a única coisa que o checkpoint da NVIDIA sabe
fazer, então é daqui que dá para partir (ver `FRASE`).

A mesa principal é a `lab_table` deles esticada 1,35x na largura (o eixo x dela,
porque ela entra girada 90°). A do coador fica em y=1,5 e não 1,2 como na da
maçã, senão encostaria na principal alargada.

A xícara e o scoop são livres; coador, chaleira, pote e tampa são fixos.

O `task_config` segue o da maçã (duas subtarefas: pegar e pôr), para o gerador de
demonstrações deles (`dexmg`) poder multiplicar episódios desta cena.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from robocasa.environments.locomanipulation.locomanip import LMSimpleEnv
from robocasa.models.scenes.lab_arena import LabArena
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneScaleConfig,
)
from robocasa.utils.scene.scene import SceneObject
from robocasa.utils.scene.success_criteria import (
    AllCriteria,
    IsInContact,
    IsUpright,
    SuccessCriteria,
)
from robocasa.utils.visuals_utls import Gradient, randomize_materials_rgba

OBJETOS = Path(__file__).resolve().parent / "objetos"
# Na forma da frase da maçã ("pick up the apple, walk left and place the apple on
# the plate."), trocando só os objetos: o modelo da NVIDIA só conhece aquela.
FRASE = "pick up the white cup, walk left and place the cup on the dripper."
LARGURA = 1.35


def _obj(nome: str) -> str:
    caminho = OBJETOS / nome / "model.xml"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe — rode gera_objetos.py antes")
    return str(caminho)


class LMCafeXicaraCoador(LMSimpleEnv, DexMGConfigHelper):
    MUJOCO_ARENA_CLS = LabArena
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    TABLE_GRADIENT = Gradient(np.array([0.68, 0.34, 0.07, 1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    LIFT_OFFSET = 0.08

    def _fixo(self, nome, x, y, z_extra=0.0, jitter=0.01, giro=(0.0, 0.0), mesa=None):
        """Objeto estático. Com `mesa`, (x, y) são relativos a ela, como o prato
        da maçã é relativo à `table_target`."""
        mesa = mesa or self.table
        ref = (dict(reference=ReferenceConfig(mesa)) if mesa is not self.table else
               dict(reference_pos=np.array([x, y, self.table.mj_obj.top_offset[2] + z_extra])))
        return SceneObject(ObjectConfig(
            name=nome, mjcf_path=_obj(nome), scale=1.0, static=True,
            sampler_config=SamplingConfig(
                x_range=np.array([-jitter, jitter]) + (x if mesa is not self.table else 0),
                y_range=np.array([-jitter, jitter]) + (y if mesa is not self.table else 0),
                rotation=np.array(giro), **ref,
            ),
        ))

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(ObjectConfig(
            name="table", mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
            scale=[LARGURA, 1.0, 1.0], static=True,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]), y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([0.5, 0, 0]), rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
            ),
        ))
        topo = self.table.mj_obj.top_offset[2]
        self.table_target = SceneObject(ObjectConfig(
            name="table_target", mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
            scale=1.0, static=True,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]), y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([0.5, 1.5, 0]), rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
            ),
        ))
        self.xicara = SceneObject(ObjectConfig(
            name="xicara", mjcf_path=_obj("xicara"), scale=1.0, static=False,
            sampler_config=SamplingConfig(
                # Onde a maçã nasce na cena deles: (0,4, 0) com a mesma faixa.
                x_range=np.array([-0.08, 0.04]), y_range=np.array([-0.08, 0.08]),
                rotation=np.array([-0.5, 0.5]),
                reference_pos=np.array([0.40, 0.0, topo]),
            ),
        ))
        self.scoop = SceneObject(ObjectConfig(
            name="scoop", mjcf_path=_obj("scoop"), scale=1.0, static=False,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.03, 0.03]), y_range=np.array([-0.03, 0.03]),
                rotation=np.array([-np.pi, np.pi]),
                reference_pos=np.array([0.40, -0.30, topo]),
            ),
        ))
        # O coador onde a maçã tem o prato: na mesa da esquerda, relativo a ela
        # (mesmos x_range/y_range do prato da `LMPnPBottleToPlate`).
        self.coador = self._fixo("coador", -0.2, 0.0, jitter=0.06, mesa=self.table_target)
        self.chaleira = self._fixo("chaleira", 0.66, 0.45, giro=(-0.4, 0.4))
        self.pote = self._fixo("pote", 0.66, -0.42)
        # A tampa em cima do pote: a referência de altura é o topo do pote.
        self.tampa = self._fixo("tampa", 0.66, -0.42, z_extra=0.10, jitter=0.0)
        return [self.table, self.table_target, self.xicara, self.scoop, self.coador,
                self.chaleira, self.pote, self.tampa]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(IsUpright(self.xicara, symmetric=False),
                           IsInContact(self.xicara, self.coador))

    def _get_instruction(self) -> str:
        return FRASE

    def get_object(self):
        return dict(
            xicara=dict(obj_name=self.xicara.mj_obj.root_body, obj_type="body"),
            coador=dict(obj_name=self.coador.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        z = self.sim.data.body_xpos[self.obj_body_id(self.xicara.mj_obj.name)][2]
        topo = self.sim.data.body_xpos[self.obj_body_id(self.table_target.mj_obj.name)][2] \
            + self.table_target.mj_obj.top_offset[2]
        return dict(obj_off_table=int(z - topo > self.LIFT_OFFSET))

    @staticmethod
    def task_config():
        comum = dict(selection_strategy="random", selection_strategy_kwargs=None,
                     action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0,
                     apply_noise_during_interpolation=False)
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(object_ref="xicara", subtask_term_signal="obj_off_table",
                                          subtask_term_offset_range=(5, 10), **comum)
        task.task_spec_0.subtask_2 = dict(object_ref="coador", subtask_term_signal=None,
                                          subtask_term_offset_range=None, **comum)
        task.task_spec_1.subtask_1 = dict(object_ref=None, subtask_term_signal=None,
                                          subtask_term_offset_range=None, **comum)
        return task.to_dict()

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            randomize_materials_rgba(rng=self.rng, mjcf_obj=self.table.mj_obj,
                                     gradient=self.TABLE_GRADIENT, linear=True)


def registra_no_gym():
    """O `sync_env` deles cria os ids do gym UMA vez, na importação, a partir do
    registro do robocasa. Esta cena nasce depois disso, então os ids dela são
    criados aqui, com a mesma função."""
    from gr00t_wbc.control.envs.robocasa import sync_env

    for robo, apelido in sync_env.GR00T_LOCOMANIP_ENVS_ROBOTS.items():
        if apelido.startswith("g1"):
            sync_env.create_gym_sync_env_class("LMCafeXicaraCoador", robo, apelido, sync_env.WBC_VERSION)


registra_no_gym()
