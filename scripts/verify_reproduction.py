"""Focused numerical checks against the recorded upstream implementation."""

import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from PIL import Image
import torch

from models.model_visual import CycleTCM, ConditionalTopKMoE, BidirectionalMoE, uncertainty_confidence
from models.model_mllm import MLLM_Adapter
from train.data import transform_views, augmentation
from train.evaluation import metrics
from utils.experiment import run_directory, write_json


def original(name):
    source = subprocess.check_output(['git', 'show', f'92c3bce:src/models/model_{name}.py'], cwd=ROOT, text=True)
    namespace = {'__name__': 'reference'}
    exec(compile(source, f'reference_{name}', 'exec'), namespace)
    return namespace['CycleTCM']


def main():
    output = run_directory(ROOT / 'outputs/verification', 'numerical')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    inputs = [torch.rand(2, 3, 224, 224, device='cuda') for _ in range(7)]
    feature = torch.rand(2, 2560, device='cuda')
    results = {}
    for name, mllm in [('visual', False), ('multimodal', True)]:
        reference = original(name)().cuda().eval()
        args = inputs + ([feature] if mllm else [])
        with torch.no_grad():
            expected = reference(*args)
        current = CycleTCM(mllm=mllm).cuda().eval()
        current.load_state_dict(reference.state_dict(), strict=True)
        with torch.no_grad():
            actual = current(*args)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), f'{name}: forward changed'
        results[name] = {'exact_forward_match': True, 'parameters': sum(p.numel() for p in current.parameters())}
        # Gate calculation checked using actual module activations captured at each boundary.
        if not mllm:
            captured = {}
            def capture(key):
                def hook(module, args, value):
                    captured[key] = value.detach().clone()
                return hook
            handles = [getattr(current, attr).register_forward_hook(capture(key)) for attr, key in
                       [('conv_whole_1_2', 'whole'), ('gate_syndrome', 'gate_logits')]]
            def local_gate(module, args, value):
                captured['local'] = args[0].detach().clone()
            handles.append(current.gate_syndrome.register_forward_hook(local_gate))
            with torch.no_grad():
                current(*args)
            gate = captured['gate_logits'].sigmoid()
            fused = gate * captured['local'] + (1 - gate) * captured['whole']
            assert torch.isfinite(fused).all() and ((gate >= 0) & (gate <= 1)).all()
            # Remove U: local classifier vector must equal explicitly reconstructed gated+position mean.
            current.uwbmoe = False
            vector = {}
            def head_input(module, args):
                vector['value'] = args[0].detach().clone()
            handle = current.bn_syndrome.register_forward_pre_hook(head_input)
            with torch.no_grad():
                current(*args)
            expected_local = (fused.flatten(2).transpose(1, 2) + current.position_syndrome).mean(dim=1)
            torch.testing.assert_close(vector['value'][:, 2048:], expected_local)
            handle.remove()
            for handle in handles:
                handle.remove()
            results['aglff_gate'] = {'reconstructed_classifier_vector_matches': True}
        del current, reference
        gc.collect()
        torch.cuda.empty_cache()
    for a, u, m in [(False,False,False), (True,False,False), (False,True,False), (False,False,True)]:
        model = CycleTCM(aglff=a, uwbmoe=u, mllm=m).cuda().eval()
        assert hasattr(model, 'gate_syndrome') == a
        assert hasattr(model, 'bidirectional_moe') == u
        assert hasattr(model, 'mllm_adapter') == m
        with torch.no_grad():
            values = model(*inputs, *([feature] if m else []))
        assert values[0].shape == (2,8) and values[1].shape == (2,5)
        results[f'modules_{a}_{u}_{m}'] = {'parameters': sum(p.numel() for p in model.parameters()), 'disabled_modules_absent': True}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    moe = ConditionalTopKMoE(d_model=4, num_experts=4, top_k=2).eval()
    x, c = torch.randn(2,3,4), torch.randn(2,3,4)
    mixed, refined, routes = moe(x,c)
    weights, indices = routes.topk(2,dim=-1)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    experts = torch.stack([expert(x) for expert in moe.experts], dim=2)
    manual = (torch.gather(experts, 2, indices.unsqueeze(-1).expand(-1,-1,-1,4)) * weights.unsqueeze(-1)).sum(2)
    torch.testing.assert_close(mixed, manual)
    torch.testing.assert_close(refined, x + manual)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2,3))
    bi = BidirectionalMoE(d_model=4).eval()
    org_mix, org_ref, _, syn_mix, syn_ref, _ = bi(x,c)
    torch.testing.assert_close(org_ref, x + org_mix)
    torch.testing.assert_close(syn_ref, c + syn_mix)
    for first, second in [(torch.zeros(2,3,4),torch.zeros(2,3,4)), (torch.ones(2,3,4),torch.ones(2,3,4)),
                          (torch.ones(2,3,4),-torch.ones(2,3,4)), (x,c)]:
        confidence = uncertainty_confidence(first, second)
        assert torch.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all()
        if torch.equal(first,second) or torch.equal(first,-second):
            torch.testing.assert_close(confidence, torch.ones_like(confidence))
        assert torch.equal(confidence, uncertainty_confidence(second,first))
    results['uwbmoe'] = {'topk_renormalization': True, 'manual_expert_mix': True, 'bidirectional_residuals': True,
                         'zero_norm_and_constant_finite': True, 'symmetric_confidence': True,
                         'constant_uncertainty_behavior': 'Confidence=1; no complementary cross-weighting'}
    image = Image.fromarray(np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8))
    transformed = transform_views([image]*7,augmentation(),True)
    assert all(torch.equal(transformed[0], value) for value in transformed)
    independent = transform_views([image]*7,augmentation(),False)
    assert any(not torch.equal(independent[0],value) for value in independent)
    results['augmentation'] = {'shared_transform_equal': True, 'compat_independent': True}
    assert MLLM_Adapter()(torch.randn(2,2560)).shape == (2,2048)
    truth = np.tile(np.array([[1,0],[0,1],[1,1],[0,0]]),(1,7))[:,:13]
    prob = np.tile(np.array([[.9,.2],[.6,.8],[.7,.4],[.2,.3]]),(1,7))[:,:13]
    measured = metrics(truth,prob)
    assert np.isclose(measured['syndrome']['acc'],.75)
    assert np.isclose(measured['syndrome']['f1'],(.8+2/3)/2)
    degenerate = metrics(np.ones((4,13)),np.full((4,13),.9))
    assert all(row['auc'] is None and row['auc_reason'] for row in degenerate['per_class'])
    logits, targets, weight = torch.randn(4,13), torch.tensor(truth,dtype=torch.float32), torch.arange(1,14,dtype=torch.float32)
    summed = sum(torch.nn.functional.binary_cross_entropy_with_logits(logits[:,section],targets[:,section],pos_weight=weight[section]) for section in [slice(0,8),slice(8,13)])
    element = torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,pos_weight=weight,reduction='none')
    torch.testing.assert_close(summed,element[:,:8].mean()+element[:,8:].mean())
    results['metrics_and_loss'] = {'task_macro_positive_f1': True,'undefined_auc_null': True,'two_task_loss_sum': True}
    write_json(output/'verification.json',results)
    print(f'PASS {output}')


if __name__ == '__main__':
    main()
