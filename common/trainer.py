import sys
sys.path.append('..')
import numpy as np
import time
import matplotlib.pyplot as plt
from common.util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x) # 100
        max_iters = data_size // batch_size # 3
        self.eval_interval = eval_interval # 20
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch): #10
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters): #3
                batch_x = x[iters * batch_size:(iters+1) * batch_size]
                batch_t = t[iters * batch_size:(iters+1) * batch_size]

                # 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                 # evaluate
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(' | epoch %d | iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iter (x {str(self.eval_interval)})')
        plt.ylabel('loss')
        plt.show()

def remove_duplicate(params, grads):
    """
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다
    :param params:
    :param grads:
    :return:
    """
    params, grads = params[:], grads[:] #copy list
    # params : 5
    while True:
        find_flag = False
        L = len(params)

        for i in range(0, L - 1): # i : 0~4
            for j in range(i + 1, L):  # j : 1~5
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flag = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flag = True
                    params.pop(j)
                    grads.pop(j)

                if find_flag: break
            if find_flag: break

        if not find_flag: break

    return params, grads
