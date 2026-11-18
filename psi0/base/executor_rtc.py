"""Executor assíncrono de blocos do Ψ0 (RTC), igual para o simulador e para o robô real.

O mundo não para enquanto o modelo pensa. A cada passo de controle (30 Hz):
  - o robô executa a próxima ação do bloco atual (com o limite de velocidade e aceleração das demos);
  - quando o bloco chega em `pedir_em` ações executadas, um pedido sai em paralelo com k = ações já executadas;
  - a resposta chega d passos depois; o bloco novo começa no instante do pedido (o servidor usa A[k:] como prefixo),
    então as d primeiras ações já passaram e são puladas;
  - se o bloco acabar antes da resposta, o robô segura o último alvo e o próximo pedido sai como primeiro (sem prefixo).

`pedir_em` se ajusta à latência medida para a resposta chegar perto de `executa` (15 no treino): pedir_em = executa - d.
O servidor (../task3/servidor_task3.py) supõe atraso de 6 passos, máximo 8; d maior funciona, mas fora do treino.
"""
import numpy as np

MAX_VEL, MAX_ACC = 2.5 / 30, 35 / 900   # rad por passo de 30 Hz: limites medidos nas demos (rodar_mujoco.py)


class ExecutorRTC:
    def __init__(self, alvo0, executa=15, tam_bloco=30, d_inicial=8):
        self.alvo = np.asarray(alvo0, float).copy(); self.passo_ant = np.zeros_like(self.alvo)
        self.executa, self.tam = executa, tam_bloco
        self.bloco = None; self.i = 0                # bloco atual e índice da próxima ação dele
        self.pendente = None                         # (k, passo do pedido) do pedido em voo
        self.d_med = float(d_inicial); self.n_pedidos = 0; self.esgotou = 0

    @property
    def pedir_em(self):
        return int(np.clip(round(self.executa - self.d_med), 2, self.tam - 2))

    def quer_pedir(self):
        """True se um pedido deve sair agora (nenhum em voo e o bloco atual chegou no ponto de pedir, ou não há bloco)."""
        return self.pendente is None and (self.bloco is None or self.i >= self.pedir_em or self.i >= self.tam)

    def pedido(self, passo):
        """Registra o pedido que sai agora. Devolve (primeiro, k) para mandar ao servidor."""
        primeiro = self.bloco is None or self.i >= self.tam
        k = 0 if primeiro else self.i
        self.pendente = (k, passo); self.n_pedidos += 1
        return primeiro, k

    def chegou(self, acoes, passo):
        """Resposta do pedido em voo. Devolve um registro da troca (d, k, salto entre o bloco velho e o novo no ponto de emenda)."""
        k, p0 = self.pendente; self.pendente = None
        d = passo - p0
        self.d_med = 0.7 * self.d_med + 0.3 * d
        novo = np.asarray(acoes, float)
        salto = None
        if self.bloco is not None and k + d < self.tam and d < len(novo):
            salto = float(np.abs(novo[d] - self.bloco[k + d]).max())
        self.bloco, self.i = novo, min(d, len(novo))
        return dict(d=d, k=k, salto_rad=salto, pedir_em=self.pedir_em)

    def proximo(self):
        """Alvo deste passo de controle: próxima ação do bloco, filtrada; sem ação disponível, segura o alvo."""
        if self.bloco is None or self.i >= len(self.bloco):
            if self.bloco is not None: self.esgotou += 1
            self.passo_ant[:] = 0; return self.alvo.copy()
        acao = self.bloco[self.i][: len(self.alvo)]; self.i += 1
        p = np.clip(acao - self.alvo, -MAX_VEL, MAX_VEL)
        p = np.clip(p, self.passo_ant - MAX_ACC, self.passo_ant + MAX_ACC)
        self.alvo = self.alvo + p; self.passo_ant = p
        return self.alvo.copy()


if __name__ == "__main__":
    # autoteste: modelo falso que manda a trajetória linear t*0.01 a partir do instante do pedido, com latência d passos
    for d in (4, 8, 12, 20):
        ex = ExecutorRTC(np.zeros(3)); voo = None; trocas = []
        for passo in range(200):
            if voo and passo >= voo[0]:
                trocas.append(ex.chegou(voo[1], passo)); voo = None
            if ex.quer_pedir():
                prim, k = ex.pedido(passo)
                assert prim == (k == 0)
                voo = (passo + d, np.array([[0.01 * (passo + j)] * 3 for j in range(30)]))
            ex.proximo()
        assert ex.n_pedidos >= 200 // 30, (d, ex.n_pedidos)
        assert all(t["d"] == d for t in trocas), d
        if d <= 12:
            assert ex.esgotou == 0, (d, ex.esgotou)
            assert all(t["salto_rad"] is None or t["salto_rad"] < 1e-9 for t in trocas), d   # trajetória contínua: emenda sem salto
            assert abs(ex.alvo[0] - 0.01 * 199) < 0.02, (d, ex.alvo)   # acompanha o relógio, não atrasa d passos
        print(f"d={d:2d}: pedidos {ex.n_pedidos}, pedir_em {ex.pedir_em}, bloco esgotou {ex.esgotou} vezes, alvo final {ex.alvo[0]:.3f} (relógio 1.990)")
    print("ok")
