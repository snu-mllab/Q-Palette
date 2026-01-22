#! /bin/bash

# example script without parallelism for Llama-3.2-1B
python gen_dev.py --model meta-llama/Llama-3.2-1B
for part in self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj mlp.up_proj mlp.gate_proj mlp.down_proj; do
    for idx in {0..15}; do
        python gen_eval_noise_kl.py --model meta-llama/Llama-3.2-1B --partition $part --group_idx $idx
    done
done
python gen_err_coeff.py --model meta-llama/Llama-3.2-1B --devset_key rnd_dev32_-1
