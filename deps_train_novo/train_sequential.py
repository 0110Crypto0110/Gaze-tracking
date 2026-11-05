#!/usr/bin/env python
"""
Treino sequencial (Continual Learning) no MPIIFaceGaze.
Baseline sem mitigação: mede o esquecimento catastrófico.

Uso (uma tarefa por execução, para evitar acúmulo de memória):
    python train_sequential.py --config configs/mpiifacegaze/resnet_50_train_0.yaml --task p00
    python train_sequential.py --config configs/mpiifacegaze/resnet_50_train_0.yaml --task p01
    ...

Ou use o script mestre (run_all_tasks.py) para chamar cada tarefa em um processo separado.
"""
import argparse
import gc
import json
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from gaze_estimation import create_loss, create_model, create_optimizer, create_scheduler
from gaze_estimation.datasets.mpiifacegaze import OnePersonDataset
from gaze_estimation.transforms import create_transform
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                   load_config, save_config, set_seeds, setup_cudnn)


def get_load_mode(config):
    if config.model.name == 'face_res50':
        return 'load_single_face'
    return 'load_multi_region'


def create_task_loaders(config, person_id, val_ratio=0.1):
    """Cria train/val loaders apenas para um participante."""
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)
    transform = create_transform(config)
    load_mode = get_load_mode(config)

    dataset = OnePersonDataset(person_id, dataset_dir, transform, load_mode)
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=config.train.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, optimizer, loss_function, train_loader, device):
    model.train()
    loss_meter = AverageMeter()
    angle_meter = AverageMeter()

    for images, gazes in train_loader:
        image = images['face'].to(device)
        left = images['left_eye'].to(device)
        right = images['right_eye'].to(device)
        gazes = gazes.to(device)

        optimizer.zero_grad()
        outputs = model(image, left, right)
        loss = loss_function(outputs, gazes)
        loss.backward()
        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()
        n = image.size(0)
        loss_meter.update(loss.item(), n)
        angle_meter.update(angle_error.item(), n)

        # Libera referências para evitar acúmulo de memória
        del image, left, right, gazes, outputs, loss, angle_error

    # Força o garbage collector do Python
    gc.collect()

    return loss_meter.avg, angle_meter.avg


def evaluate_on_task(model, config, person_id, device):
    """Avalia o modelo em um participante específico (3000 amostras)."""
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)
    transform = create_transform(config)
    load_mode = get_load_mode(config)

    dataset = OnePersonDataset(person_id, dataset_dir, transform, load_mode)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model.eval()
    angle_meter = AverageMeter()
    with torch.no_grad():
        for images, gazes in loader:
            image = images['face'].to(device)
            left = images['left_eye'].to(device)
            right = images['right_eye'].to(device)
            gazes = gazes.to(device)
            outputs = model(image, left, right)
            angle_error = compute_angle_error(outputs, gazes).mean()
            angle_meter.update(angle_error.item(), image.size(0))

            # Libera referências
            del image, left, right, gazes, outputs, angle_error

    gc.collect()
    return angle_meter.avg


def compute_average_forgetting(perf_matrix):
    """
    perf_matrix[i][j] = erro na tarefa i APÓS treinar até a tarefa j.
    AF = média de (erro_final_em_i - erro_após_treinar_em_i)
    """
    n = perf_matrix.shape[0]
    forgetting = 0.0
    for i in range(n - 1):
        best = perf_matrix[i][i]
        final = perf_matrix[i][n - 1]
        forgetting += final - best
    return forgetting / (n - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True,
                        help='Participante a treinar (ex: p00, p01, ..., p14)')
    parser.add_argument('--force', action='store_true',
                        help='Re-treina mesmo se o checkpoint da tarefa já existir')
    args = parser.parse_args()

    # Carrega config manualmente (o load_config lê de --config global)
    import sys
    sys.argv = [sys.argv[0], '--config', args.config]
    config = load_config()
    set_seeds(config.train.seed)
    setup_cudnn(config)

    device = torch.device(config.device)
    print(f'Device: {device}')

    output_dir = pathlib.Path('experiments/mpiifacegaze/continual/baseline_run')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    # Verifica se a tarefa já foi treinada
    checkpoint_file = output_dir / f'after_task_{args.task}.pth'
    if checkpoint_file.exists() and not args.force:
        print(f'Checkpoint para {args.task} já existe. Pulando (use --force para re-treinar).')
        return

    task = args.task
    task_idx = int(task[1:])  # p00 -> 0, p01 -> 1, ...

    # Carrega a matriz de performance parcial (se existir)
    perf_file = output_dir / 'perf_matrix.npy'
    if perf_file.exists():
        perf_matrix = np.load(perf_file)
        print(f'Matriz de performance carregada: {perf_matrix.shape}')
    else:
        perf_matrix = np.full((15, 15), np.nan)

    # Cria o modelo
    model = create_model(config)

    # Carrega o checkpoint da tarefa anterior (se existir)
    if task_idx > 0:
        prev_task = f'p{task_idx - 1:02d}'
        prev_checkpoint = output_dir / f'after_task_{prev_task}.pth'
        if prev_checkpoint.exists():
            print(f'Carregando checkpoint anterior: {prev_checkpoint}')
            ckpt = torch.load(prev_checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['model'])
        else:
            print(f'AVISO: checkpoint anterior {prev_checkpoint} não encontrado. Iniciando do zero.')

    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)

    print(f'\n{"="*60}')
    print(f'Tarefa {task_idx+1}/15: treinando em {task}')
    print(f'{"="*60}')
    start = time.time()

    train_loader, _ = create_task_loaders(config, task)

    for epoch in range(1, config.scheduler.epochs + 1):
        loss_avg, angle_avg = train_one_epoch(
            model, optimizer, loss_function, train_loader, device
        )
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f'  Época {epoch:2d}: loss={loss_avg:.4f}, angle={angle_avg:.2f}°')

    elapsed = time.time() - start
    print(f'  Tempo de treino: {elapsed/60:.1f} min')

    # Avalia em TODAS as tarefas vistas até agora
    print(f'  Avaliando em {task_idx+1} tarefa(s) vista(s):')
    for i in range(task_idx + 1):
        error = evaluate_on_task(model, config, f'p{i:02d}', device)
        perf_matrix[i][task_idx] = error
        print(f'    p{i:02d}: {error:.2f}°')

    # Salva checkpoint da tarefa
    torch.save({
        'model': model.state_dict(),
        'task_idx': task_idx,
        'task_name': task,
    }, checkpoint_file)
    print(f'  Checkpoint salvo: {checkpoint_file}')

    # Salva matriz parcial
    np.save(perf_file, perf_matrix)

    # Se for a última tarefa, calcula e salva os resultados finais
    if task_idx == 14:
        print(f'\n{"="*60}')
        print('RESULTADO FINAL — Baseline (sem mitigação)')
        print(f'{"="*60}')

        final_errors = perf_matrix[:, 14]
        avg_acc = np.nanmean(final_errors)
        af = compute_average_forgetting(perf_matrix)

        print(f'ACC final (média de erro nas 15 tarefas): {avg_acc:.2f}°')
        print(f'Average Forgetting (AF): {af:.2f}°')

        results = {
            'perf_matrix': perf_matrix.tolist(),
            'avg_accuracy': float(avg_acc),
            'average_forgetting': float(af),
            'tasks': [f'p{i:02d}' for i in range(15)],
        }
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Resultados salvos em: {output_dir}')


if __name__ == '__main__':
    main()