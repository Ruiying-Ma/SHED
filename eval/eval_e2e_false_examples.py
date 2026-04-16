
true_evidence = '''This marked the 65th consecutive year of dividend increases for 3M'''
import json

with open("eval_e2e_false_examples.json", 'r') as file:
    false_examples = json.load(file)
raptor_ctx: str = false_examples["raptor_ctx"]
shed_ctx: str = false_examples["shed_ctx"]

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
raptor_token_count = len(tokenizer.encode(raptor_ctx))
shed_token_count = len(tokenizer.encode(shed_ctx))

print(f"Raptor context token count: {raptor_token_count}")
print(f"Shed context token count: {shed_token_count}")

assert true_evidence in raptor_ctx, "True evidence not found in Raptor context"
assert true_evidence in shed_ctx, "True evidence not found in Shed context"

raptor_idx = raptor_ctx.find(true_evidence)
shed_idx = shed_ctx.find(true_evidence)

print(f"Raptor index: {raptor_idx / len(raptor_ctx):.2%} into context")
print(f"Shed index: {shed_idx / len(shed_ctx):.2%} into context")