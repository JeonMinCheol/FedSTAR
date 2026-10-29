"""CUDA selection before importing torch, followed by explicit runtime binding.

Keep this module free of torch/numpy/framework imports: main.py calls the
bootstrap before importing any library that could inspect or initialize CUDA.
"""

import argparse
import os
import re


def gpu_ids(value):
    value = str(value).strip()
    tokens = [token.strip() for token in value.split(',')]
    if (not tokens or any(not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/.-]+)', token)
                          for token in tokens) or len(set(tokens)) != len(tokens)):
        raise argparse.ArgumentTypeError(
            'GPU ID must be a nonnegative index, GPU/MIG UUID, or a comma-separated list '
            'of distinct IDs; use --device cpu for CPU execution.'
        )
    return ','.join(tokens)


def bootstrap_cuda_environment(argv=None, environ=None):
    environment = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('-dev', '--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('-did', '--device_id', type=gpu_ids, default=None)
    args, _ = parser.parse_known_args(argv)
    inherited = environment.get('CUDA_VISIBLE_DEVICES')
    if args.device == 'cpu':
        # Legacy checkpoint loaders must not move CPU runs to a visible GPU.
        visible = ''
    elif args.device_id is not None:
        visible = args.device_id
    else:
        # Respect a shell/scheduler mask when no explicit -did is provided.
        visible = inherited if inherited is not None else '0'
    environment['CUDA_VISIBLE_DEVICES'] = visible
    if args.device_id is not None:
        # Explicit numeric -did refers to the PCI-bus ordering used by nvidia-smi.
        environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    else:
        environment.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    return {
        'visible': visible, 'inherited': inherited,
        'explicit': args.device_id is not None,
        'order': environment['CUDA_DEVICE_ORDER'],
    }


def configure_runtime_device(args, torch, selection):
    """Bind logical device 0 in the mask fixed by bootstrap; never remask CUDA."""
    visible = selection['visible']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != visible:
        raise RuntimeError('CUDA_VISIBLE_DEVICES changed after startup; restart with the desired -did')
    if args.device == 'cpu':
        print(f'[Device] PID={os.getpid()} device=cpu CUDA_VISIBLE_DEVICES=<empty>', flush=True)
        return
    args.device_id = visible
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            f'CUDA unavailable for CUDA_VISIBLE_DEVICES={visible!r}, '
            f'CUDA_DEVICE_ORDER={selection["order"]}, PyTorch CUDA build={torch.version.cuda!r}. '
            'Check that the selected GPU exists and is accessible to this process; '
            'check nvidia-smi on the host. Use --device cpu for CPU execution.'
        )
    count = torch.cuda.device_count()
    if selection['explicit'] and count != len(visible.split(',')):
        raise RuntimeError(f'Requested GPUs {visible!r}, but CUDA exposes {count} device(s); '
                           'check -did indices/UUIDs and process GPU access')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid = getattr(properties, 'uuid', '<unavailable>')
    gpu = visible.split(',')[0]
    print(f'[Device] PID={os.getpid()} CUDA_DEVICE_ORDER={selection["order"]} '
          f'CUDA_VISIBLE_DEVICES={visible!r}', flush=True)
    print(f'[Device] selected GPU {gpu} -> logical cuda:0; '
          f'name={properties.name}; UUID={uuid}', flush=True)
    if selection['explicit'] and selection['inherited'] is not None and selection['inherited'] != visible:
        print(f'[Device] explicit -did replaced inherited CUDA_VISIBLE_DEVICES='
              f'{selection["inherited"]!r}', flush=True)
