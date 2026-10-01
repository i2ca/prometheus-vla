"""Pesos por junta para o IK preferir o movimento que uma pessoa faria.

Medido no levantamento com a pega de referencia: a mao anda 13 cm ate a alca
com o cotovelo variando 0,2 grau e o umero girando 41 graus; na subida o punho
torce 24 graus. O IK regularizava todas as juntas por igual, entao girar o
umero (com o cotovelo ja dobrado) era o jeito mais barato de levar a mao.
Uma pessoa alcanca e levanta flexionando ombro e cotovelo, quase sem girar o
umero, torcer o punho ou mexer o tronco. Os pesos multiplicam o termo de
regularizacao (x - referencia): maior = junta mais cara de mexer.
"""
import numpy as np

PESOS = {
    'waist_yaw_joint': 3., 'waist_roll_joint': 5., 'waist_pitch_joint': 5.,
    'shoulder_pitch_joint': 1., 'shoulder_roll_joint': 1.5, 'shoulder_yaw_joint': 4.,
    'elbow_joint': .6,
    'wrist_roll_joint': 3., 'wrist_pitch_joint': 1.5, 'wrist_yaw_joint': 3.,
}


def pesos(nomes):
    """Vetor de pesos na ordem de `nomes`; juntas sem regra (dedos) pesam 1.

    Usado de duas formas: multiplicando a regularizacao e como x_scale=1/pesos
    no least_squares, que e o que de fato molda o passo no espaco nulo.
    """
    saida = []
    for n in nomes:
        chave = n.replace('left_', '').replace('right_', '')
        saida.append(PESOS.get(chave, 1.))
    return np.array(saida)
