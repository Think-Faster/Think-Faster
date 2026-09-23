"""Шаг 6. Нейросеть по часовым рядам объекта (Ф5-1, сравнение с бустингом).

Вход — последние 7 суток часовых счётчиков объекта и среднее по его коллектору (work/seq.npz),
плюс статика: состав датчиков, календарь, эпизоды за 30 и 90 суток, часы с прошлого эпизода.
Сеть — причинные свёртки с растущим шагом (охват 255 ч), выход — 6 логитов, по одному на тип.

Ряды целиком лежат в памяти видеокарты (~0,8 ГБ в fp16), окна режутся индексами прямо на GPU,
поэтому DataLoader и процессы-загрузчики не нужны: в btc они разгружали CPU-подготовку батчей,
здесь её нет. Обучение в bf16. Строки обучения/проверки/теста — те же, что в витрине, прогнозы
пишутся рядом с прогнозами бустинга (work/runs/<ветка>/preds/tcn_*), чтобы их можно было смешивать.

    python seqmodel.py --epochs 12
"""
import argparse
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import config
import metrics
from features import CAP, IDX, ONSETS, calendar, since_last
from train import FEAT, SPLITS

L = 168


class Block(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.pad, 0)))          # причинная свёртка: только прошлое
        y = self.mix(self.drop(F.gelu(self.norm(y))))
        return x + y


class Net(nn.Module):
    def __init__(self, c_in: int, c_static: int, n_out: int, width: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(c_in, width, 1)
        self.blocks = nn.Sequential(*[Block(width, d, dropout) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Sequential(nn.Linear(2 * width + c_static, 256), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(256, n_out))

    def forward(self, x, s):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([h[:, :, -1], h.mean(-1), s], 1))


def index(years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    df = pl.scan_parquet([FEAT / f'{y}.parquet' for y in years]).select('object_id', 'h').collect()
    return df['object_id'].to_numpy(), df['h'].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--per-epoch', type=int, default=400_000)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--branch', default='main', choices=list(SPLITS))
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = 'cuda'
    H, types = args.horizon, config.TYPES
    t = time.time()

    with np.load(config.WORK / 'seq.npz') as f:
        z = {k: f[k] for k in f.files}                               # npz читает массив заново при каждом обращении
    objects = z['objects']
    oi = {int(o): i for i, o in enumerate(objects)}
    base_np = z.pop('base')
    since = np.stack([np.stack([since_last(np.expm1(base_np[o, :, IDX[f'onset_{tp}']].astype(np.float32)))
                                for tp in types], 1) for o in range(len(objects))])
    base = torch.from_numpy(base_np).to(dev)                         # объекты × часы × каналы, fp16
    del base_np
    collectors = z['collectors']
    uniq = np.unique(collectors)
    o2c = torch.tensor(np.searchsorted(uniq, collectors), device=dev)
    coll = torch.stack([base[torch.from_numpy(collectors == c).to(dev)].float().mean(0) for c in uniq]).half()
    # долгая память: эпизоды за 30 и 90 суток и часы с прошлого эпизода — окно 7 суток их не видит
    csum = torch.cumsum(torch.expm1(base[:, :, [IDX[k] for k in ONSETS]].float()), 1)
    since = torch.from_numpy(np.log1p(since)).to(dev).half()
    cal = calendar()
    cal = torch.from_numpy(np.stack([cal['hour'] / 23, cal['dow'] / 6, cal['month'] / 12, cal['doy_sin'],
                                     cal['doy_cos'], cal['holiday'], cal['long_holiday'], cal['may9'],
                                     cal['days_to_holiday'] / 60], 1).astype(np.float32)).to(dev)
    comp = torch.from_numpy(np.log1p(z['composition'])).to(dev)
    nxt = torch.from_numpy(np.stack([z[f'next_{tp}'] for tp in types], -1)).to(dev)   # int16
    y_all = (nxt <= H)
    print(f'ряды на GPU {tuple(base.shape)} за {time.time() - t:.0f} с, '
          f'{torch.cuda.memory_allocated() / 2 ** 30:.2f} ГБ', flush=True)

    def batch(o: torch.Tensor, h: torch.Tensor):
        hours = (h[:, None] - torch.arange(L - 1, -1, -1, device=dev)[None]).clamp(min=0)
        x = torch.cat([base[o[:, None], hours], coll[o2c[o][:, None], hours]], -1)
        long = [(csum[o, h] - csum[o, (h - w).clamp(min=0)]).log1p() for w in (720, CAP)]
        s = torch.cat([comp[o], cal[h], since[o, h].float()] + long, 1)
        return x, s

    def to_idx(obj, h):
        return (torch.tensor([oi[int(x)] for x in obj], device=dev), torch.tensor(h, device=dev, dtype=torch.long))

    tr_o, tr_h = to_idx(*index(SPLITS[args.branch][0]))
    out_dir = config.WORK / 'runs' / f'{args.branch}_h{H}'
    idx = {s: np.load(out_dir / 'preds' / f'index_{s}.npz') for s in ('val', 'test')}
    ev = {s: to_idx(idx[s]['object_id'], idx[s]['h']) for s in idx}

    x0, s0 = batch(tr_o[:2], tr_h[:2])
    net = Net(x0.shape[-1], s0.shape[-1], len(types), args.width, args.dropout).to(dev)
    p = y_all[tr_o, tr_h].float().mean(0)
    pos_weight = ((1 - p) / p.clamp(min=1e-4)).sqrt().clamp(1, 10)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * (args.per_epoch // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    print(f'параметров {sum(q.numel() for q in net.parameters()):,}, обучающих строк {len(tr_o):,}', flush=True)

    @torch.no_grad()
    def predict(o_all, h_all) -> np.ndarray:
        net.eval()
        out = []
        for i in range(0, len(o_all), 4096):
            x, s = batch(o_all[i:i + 4096], h_all[i:i + 4096])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(torch.sigmoid(net(x, s).float()))
        net.train()
        return torch.cat(out).cpu().numpy()

    best, best_state = -1.0, None
    vo, vh = ev['val']
    yv = y_all[vo, vh].cpu().numpy()
    for epoch in range(args.epochs):
        t1 = time.time()
        perm = torch.from_numpy(rng.choice(len(tr_o), args.per_epoch, replace=False)).to(dev)
        total = torch.zeros((), device=dev)
        for i in range(0, args.per_epoch - args.batch + 1, args.batch):
            j = perm[i:i + args.batch]
            o, h = tr_o[j], tr_h[j]
            x, s = batch(o, h)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = net(x, s)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y_all[o, h].float(), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.detach()
        pv = predict(vo[::3], vh[::3])
        aps = [average_precision_score(yv[::3, k], pv[:, k]) for k in range(len(types))]
        score = float(np.mean(aps))
        print(f'эпоха {epoch + 1}: loss {total.item() / (args.per_epoch // args.batch):.4f}, PR-AUC на проверке '
              + ' '.join(f'{tp} {a:.3f}' for tp, a in zip(types, aps)) + f' | среднее {score:.3f}, '
              f'{time.time() - t1:.0f} с, пик {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} ГБ', flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    (out_dir / 'models').mkdir(parents=True, exist_ok=True)
    # Помимо весов кладём архитектуру: без c_in/c_static/width число сеть в проде не собрать
    # (retro.load_nets читает ровно эти ключи, INTEGRATION §1.4)
    torch.save({'state_dict': best_state,
                'c_in': x0.shape[-1], 'c_static': s0.shape[-1],
                'n_out': len(types), 'width': args.width, 'dropout': args.dropout},
               out_dir / 'models' / 'tcn.pt')

    preds = {s: predict(*ev[s]) for s in ev}
    report = {}
    cap = int(z['next_' + types[0]].max())
    for k, tp in enumerate(types):
        pv, ps = preds['val'][:, k], preds['test'][:, k]
        np.save(out_dir / 'preds' / f'tcn_{tp}_val.npy', pv)
        np.save(out_dir / 'preds' / f'tcn_{tp}_test.npy', ps)
        yv_k = (z[f'next_{tp}'][ev['val'][0].cpu().numpy(), idx['val']['h']] <= H).astype(np.float32)
        thr = metrics.best_threshold(yv_k, pv)
        row = {}
        for split, pr in (('val', pv), ('test', ps)):
            obj, hh = idx[split]['object_id'], idx[split]['h']
            oo = ev[split][0].cpu().numpy()
            for target in ('', '_prim', '_conf'):
                row[f'{split}{target}'] = metrics.evaluate(obj, hh, z[f'next_{tp}{target}'][oo, hh], pr, thr, H, cap)
        report[tp] = {'scores': {'tcn': row}}
        s_, sp = row['test'], row['test_prim']
        print(f'  {tp:9s} tcn test PR-AUC {s_["pr_auc"]:.3f} P {s_["precision"]:.3f} R(эп) {s_["recall_episodes"]:.3f}'
              f' | первичные PR-AUC {sp["pr_auc"]:.3f}', flush=True)
    (out_dir / 'report_tcn.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'готово за {time.time() - t:.0f} с')


if __name__ == '__main__':
    main()
