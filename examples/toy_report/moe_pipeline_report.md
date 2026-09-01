# Transformed model verification (field engine)

Date: 2026-08-31 18:50:11  
Stage: base MoE -> compact field (no explicit expert weights) -> loading from the artifact and verification.  
Base: ppl **8.12** (ce 2.095), eval: 20000 val tokens, 120 chunks, protocol identical to the transformation.  
Deploy: the field is stored in fp16, computed in fp32 (de-quantization at load).

## Metrics on the transformed model

| r | field, MB (claimed/actual) | expert compression | KL, bits/token | ppl | Δppl | fp32-fit KL (CSV) | gen. match |
|---|---|---|---|---|---|---|---|
| 16 | 0.48 / 0.48 | x6.6 | 0.036 | 8.36 | +2.9% | - | 36% |
| 32 | 0.56 / 0.56 | x5.6 | 0.029 | 8.32 | +2.5% | - | 31% |
| 8 | 0.44 / 0.44 | x7.2 | 0.182 | 9.25 | +13.9% | 0.182 | 38% |

Full size of the explicit experts (fp16): 3.15 MB; the r=16 artifact file: 1.60 MB (of which backbone 1.13 MB - the shared part of the model; in a real scenario it already exists and is not counted as compression).

## Generation (greedy, same seed)

**Base model (explicit experts):**

```text
 wind of the north,
To do me bust the the the the there ton there to ther tour ther to ther to the the the the the the the the there ther tour t there to ther the to the the the the the the se
```
**Transformed (field r=16, fp16 artifact):**

```text
 wind of the north,
To do me bust the the thee there there there therer theat ther thee the theer an tres tone t the treat the the see t to thee t thee st treate ten teer teer teer ther treat 
```
**Transformed (field r=32, fp16 artifact):**

```text
 wind of the north,
To do me bust the the thee there there there therer therer theat ther and the the the the see all t the the see alll t to thee se t to the the see ar treat tees and to the 
```
**Transformed (field r=8, fp16 artifact):**

```text
 wind of the north,
To do me bust the the thee thee there thee there there there ther thee ther stheer st theeer t theer ther theer ser t theeer t theer t theer t theee st thee st st theeer se
```
## Conclusion

The transformed model deploys from the artifact without explicit expert weights and works: the best rank r=32 gives KL 0.029 bits/token and Δppl +2.5% at expert-memory compression x5.6. The price of fp16 storage vs the fp32 fit: no CSV data to cross-check.
