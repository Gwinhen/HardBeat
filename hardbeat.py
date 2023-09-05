import numpy as np
import torch
import sys
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision.utils import save_image
from util import decision_function


def hardbeat(model, inputs, normalize, target, size=7, pos=(14, 14),
             query_limit=50000):

    input_size = inputs.shape[2]
    region = True if input_size > 32 else False
    qlimit = query_limit / len(inputs)

    fmax = - float('inf')
    init_mask = None
    init_pattern = None
    init_predict = None
    noise_shape = [100] + list(inputs[0].shape)
    if region:
        noise_shape[2:] = [32, 32]

    ### calculate # iterations for each stage ###
    stage1_total = int(qlimit * 0.1)
    stage1_iter  = int(stage1_total / 2)
    if stage1_total > 40:
        stage1_iter  = int(stage1_iter * 0.6)
    elif stage1_total > 30:
        stage1_iter  = int(stage1_iter * 0.7)
    elif stage1_total > 25:
        stage1_iter  = int(stage1_iter * 0.8)
    else:
        stage1_iter
    stage1_trial = 2

    stage2_total = int(qlimit * 0.6)
    stage2_trial = 30 if stage2_total > 50 else 20
    stage2_iter  = int(stage2_total / stage2_trial)

    stage3_total = int(qlimit * 0.3)
    stage3_trial = 10
    stage3_iter  = int(stage3_total / stage3_trial)
    stage3_iter  = int(np.ceil((qlimit - stage1_total - stage2_total)\
                        / stage3_trial))

    query_count = 0


    #===================== stage 1 =====================#
    best_xi = int(np.ceil((input_size - size) / 2 + 1))
    best_yi = best_xi

    side = input_size // 5

    print('-'*50)
    print('Initial search...')
    print('-'*50)
    for i in range(stage1_iter):
        ### initialize location ###
        mask = torch.rand(inputs[0, 0].shape)

        if i == 0:
            xi, yi = best_xi, best_yi
        else:
            stride = 1 if torch.rand(1)[0] < 0.5 else 2
            if region:
                stride *= 4

            remain = input_size - size + 1 - side * 2

            xi = torch.randint(remain // stride, (1,))[0] * stride + side
            yi = torch.randint(remain // stride, (1,))[0] * stride + side

        if i > 10:
            local = 2
            if region:
                local = 8
            xi = best_xi + np.random.randint(-local, local)
            yi = best_yi + np.random.randint(-local, local)

        ### initialize mask ###
        mask = torch.zeros(inputs[0, 0].shape)
        mask[xi:xi+size, yi:yi+size] = 1

        mask = torch.unsqueeze(mask, axis=0).repeat(3, 1, 1)

        noise_shape[0] = stage1_trial
        noise = torch.rand(noise_shape)
        noise = 8 * (noise - 0.5)
        noise = T.Resize(input_size)(noise)

        ### find pattern ###
        favg = 0
        preds = []
        for j in range(len(inputs)):
            perturbed = (1 - mask) * inputs[j] + mask * noise
            perturbed = torch.clamp(perturbed, 0, 1)

            decisions = decision_function(model, perturbed, normalize, target)
            query_count += len(perturbed)
            for _ in range(len(inputs.shape[1:])):
                decisions = decisions.unsqueeze(dim=1)
            fval = 2 * decisions.type(torch.FloatTensor) - 1.0
            preds.append(fval.squeeze())
            favg += torch.mean(fval)

        favg = favg / len(inputs)
        if favg > fmax:
            best_xi = xi
            best_yi = yi
            fmax = favg
            init_mask = mask
            init_pattern = noise
            init_predict = preds

        sys.stdout.write(f'\r{i}: {fmax:.4f}')

    print()
    print('max flip rate:\t', fmax.numpy())
    print('pixel num:\t', torch.count_nonzero(init_mask).numpy() / 3)

    pred = torch.sum(torch.stack(init_predict), dim=0)
    idx = torch.argmax(pred)
    init_pattern = init_pattern[idx]

    perturbed = (1 - init_mask) * inputs + init_mask * init_pattern
    perturbed = torch.clamp(perturbed, 0, 1)
    decisions = decision_function(model, perturbed, normalize, target)
    print('stage1 asr:\t',
            decisions.type(torch.FloatTensor).sum().numpy() / len(decisions))
    print('# queries:\t', query_count)

    ### improve pattern ###
    print('-'*30)
    favg = 0
    for i in range(stage1_total - stage1_iter * stage1_trial):
        noise = torch.rand(noise_shape[1:])
        noise = 8 * (noise - 0.5)
        noise = T.Resize(input_size)(noise)

        perturbed = (1 - init_mask) * inputs + init_mask * (init_pattern + noise)
        perturbed = torch.clamp(perturbed, 0, 1)

        decisions = decision_function(model, perturbed, normalize, target)
        query_count += len(perturbed)
        fval = 2 * decisions.type(torch.FloatTensor) - 1.0
        favg = torch.mean(fval)

        if favg > fmax:
            fmax = favg
            init_pattern = init_pattern + noise

    perturbed = (1 - init_mask) * inputs + init_mask * init_pattern
    perturbed = torch.clamp(perturbed, 0, 1)
    decisions = decision_function(model, perturbed, normalize, target)
    print('stage1.1 asr:\t',
            decisions.type(torch.FloatTensor).sum().numpy() / len(decisions))
    print('# queries:\t', query_count)


    #===================== stage 2 =====================#
    mask = init_mask
    pattern = torch.clamp(init_mask * init_pattern, 0 ,1)

    lr = 0.3 if region else 0.8
    eps = 1e-10
    grad_momentum = 0
    full_matrix = 0
    beta1 = 0.5
    beta2 = 0.5

    signs = None
    favg_diff = 0
    last_grad = 0
    last_favg = fmax
    last_fval = []
    hist_favg = [fmax]
    hist_pattern = [pattern]

    search_iter = stage2_iter + stage3_iter
    nn_graph = np.zeros((search_iter+1, search_iter+1))

    print('-'*50)
    print('Gradient search...')
    print('-'*50)
    for i in range(stage2_iter):
        ### select local minima ###
        num = len(hist_favg)
        k = np.minimum(4, num)
        topk = torch.topk(torch.stack(hist_favg), k)

        topk_value = topk[0].numpy()
        topk_prob = np.exp(topk_value * 5)
        topk_prob /= np.sum(topk_prob)

        topk_index = topk[1].numpy()

        selected = np.random.choice(topk_index, 1, replace=False, p=topk_prob)[0]

        prob = (hist_favg[selected] + 1) / (last_favg + 1 + eps)
        if torch.rand(1)[0] <= prob:
            pattern = hist_pattern[selected]

        noise_shape[0] = stage2_trial
        noise = torch.rand(noise_shape)
        noise = 2.0 * (noise - 0.5)
        noise = T.Resize(input_size)(noise)

        ### normalize perturbation ###
        if favg_diff < 0:
            scale = np.maximum(0.9 + favg_diff // 0.2, 0.8)
            noise *= torch.exp(noise.abs() - 1) * scale

        ### update gradient sign ###
        if signs is not None:
            if favg_diff > 0:
                n = int(favg_diff / 0.1) - 1
            else:
                n = - int(favg_diff / 0.1)
                signs = - signs

            n = min(stage2_trial - i, n)
            if n > 0:
                idx = np.random.choice(np.arange(i, stage2_trial), n, replace=False)
                noise[idx] = noise[idx].abs() * signs

        trigger = pattern + noise

        ### interpolation of local minima ###
        if i > 3:
            for t in range(np.minimum(i, stage2_trial)):
                choice1 = np.random.choice(topk_index, 1)[0]
                prob = nn_graph[choice1][:num]
                prob = prob * (np.array(hist_favg) + 1)
                if np.sum(prob) == 0:
                    prob[...] = 1
                prob = prob / np.sum(prob)
                choice2 = np.random.choice(num, 1, p=prob)[0]

                eta = torch.rand(1)[0]
                new_pattern = eta * hist_pattern[choice1]\
                                + (1 - eta) * hist_pattern[choice2]
                trigger[t] = new_pattern + noise[t] * 0.05

        trigger = mask * trigger

        ### gradient estimation ###
        favg = 0
        grads = []
        for j in range(len(inputs)):
            perturbed = (1 - mask) * inputs[j] + trigger
            perturbed = torch.clamp(perturbed, 0, 1)
            rv = perturbed - inputs[j] - mask * pattern

            decisions = decision_function(model, perturbed, normalize, target)
            query_count += len(perturbed)
            for _ in range(len(inputs.shape[1:])):
                decisions = decisions.unsqueeze(dim=1)
            fval = 2 * decisions.type(torch.FloatTensor) - 1.0
            fmean = torch.mean(fval)
            favg += fmean

            ### check asr improvement compared to the last time ###
            if len(last_fval) < len(inputs):
                diff = fmean - last_favg
            else:
                diff = fmean - last_fval[j]

            ### rescale gradient magnitude for different samples ###
            if diff > 0:
                if fmean < last_favg:
                    diff = 20
                else:
                    diff = torch.exp(diff)
                    if fmean == 1:
                        diff *= 0.2
            elif diff < 0:
                diff_scale = np.minimum(fmean + 1.1, 0.6)
                diff = torch.log((diff + 3)) * diff_scale

            if len(last_fval) < len(inputs):
                last_fval.append(fmean)
            else:
                last_fval[j] = fmean

            if fmean == 1.0:
                gradf = torch.mean(rv, dim=0)
            elif fmean == -1.0:
                gradf = - torch.mean(rv, dim=0)
            else:
                fval -= fmean
                gradf = torch.mean(fval * rv, dim=0)

            gradf = gradf * diff / torch.linalg.norm(gradf)
            grads.append(gradf)

        favg = favg / len(inputs)
        favg_diff = favg - last_favg

        gradf = torch.mean(torch.stack(grads), dim=0)

        ### record gradient sign for the next iteration ###
        if (favg > last_favg and favg > -0.5)\
                or (favg < last_favg and last_favg > -0.7):
            signs = torch.sign(gradf)
        else:
            signs = None

        if region:
            gradf = gradf[mask > 0]
        gradf_flat = gradf.flatten()

        ### adam optimizer ###
        if i == 0:
            grad_momentum = gradf
            full_matrix   = torch.outer(gradf_flat, gradf_flat)
        else:
            grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
            full_matrix   = beta2 * full_matrix\
                            + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)

        grad_momentum /= (1 - beta1 ** (i + 1))
        full_matrix   /= (1 - beta2 ** (i + 1))
        factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
        gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)

        if favg < last_favg:
            gradf *= torch.min(0.1 / (last_favg - favg), torch.tensor(0.5))

        if region:
            pattern[mask > 0] = pattern[mask > 0] + gradf * lr
        else:
            pattern = pattern + gradf * lr

        ### record neighboring triggers ###
        for t in range(num):
            p1 = hist_pattern[t].flatten()
            p2 = (pattern * mask).flatten()
            sim = (F.cosine_similarity(p1, p2, dim=0) + 1) / 2
            nn_graph[t, i+1] = sim
            nn_graph[i+1, t] = sim

        hist_favg.append(favg)
        hist_pattern.append(pattern)

        last_favg = favg

        perturbed = (1 - mask) * inputs + mask * pattern
        perturbed = torch.clamp(perturbed, 0, 1)

        sys.stdout.write(f'\r{i}: {favg:.4f}, '\
                         + f'grad: ({gradf.min().numpy():.4f}, '\
                         + f'{gradf.max().numpy():.4f}), '\
                         + f'pattern: ({pattern.min().numpy():.4f}, '\
                         + f'{pattern.max().numpy():.4f})   ')

    print()


    #===================== stage 3 =====================#
    print('-'*50)
    print('Neighbourhood search...')
    print('-'*50)
    trigger = trigger[:stage3_trial]
    for i in range(stage2_iter, search_iter):
        num = len(hist_favg)
        k = np.minimum(4, num)
        topk = torch.topk(torch.stack(hist_favg), k)

        topk_index = topk[1].numpy()

        noise_shape[0] = stage3_trial
        noise = torch.rand(noise_shape)
        noise = 1.0 * (noise - 0.5)
        noise = T.Resize(input_size)(noise)

        trigger = pattern + noise

        ### interpolation of local minima ###
        for t in range(stage3_trial):
            choice1 = np.random.choice(topk_index, 1)[0]
            prob = nn_graph[choice1][:num]
            prob = prob * (np.array(hist_favg) + 1)
            if np.sum(prob) == 0:
                prob[...] = 1
            prob = prob / np.sum(prob)
            choice2 = np.random.choice(num, 1, p=prob)[0]

            eta = torch.rand(1)[0]
            new_pattern = eta * hist_pattern[choice1]\
                            + (1 - eta) * hist_pattern[choice2]
            trigger[t] = new_pattern + noise[t] * 0.1
        trigger = mask * trigger

        ### search on the ridge ###
        best_asr = favg
        best_ptn = pattern
        for j in range(len(trigger)):
            perturbed = (1 - mask) * inputs + trigger[j]
            perturbed = torch.clamp(perturbed, 0, 1)

            decisions = decision_function(model, perturbed, normalize, target)
            query_count += len(perturbed)
            fval = 2 * decisions.type(torch.FloatTensor) - 1.0
            asr = torch.mean(fval)
            if asr > best_asr:
                best_asr = asr
                best_ptn = trigger[j]

        favg = best_asr
        pattern = best_ptn

        for t in range(num):
            p1 = hist_pattern[t].flatten()
            p2 = (pattern * mask).flatten()
            sim = (F.cosine_similarity(p1, p2, dim=0) + 1) / 2
            nn_graph[t, i+1] = sim
            nn_graph[i+1, t] = sim

        hist_favg.append(favg)
        hist_pattern.append(pattern)

        perturbed = (1 - mask) * inputs + mask * pattern
        perturbed = torch.clamp(perturbed, 0, 1)

        sys.stdout.write(f'\r{i}: {favg:.4f}, '\
                         + f'pattern: ({pattern.min().numpy():.4f}, '\
                         + f'{pattern.max().numpy():.4f})   ')

        if query_count >= query_limit:
            break

    print()


    decisions = decision_function(model, perturbed, normalize, target)
    print('stage2 asr:\t',
            decisions.type(torch.FloatTensor).sum().numpy() / len(decisions))

    print('-'*50)
    print('total queries:\t', query_count)
    print('-'*50)

    return perturbed, mask, pattern
