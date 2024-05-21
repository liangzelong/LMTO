## LMTO
Repository for "*LMTO: Large Model Guided Topology
Optimization for Conceptual Design
Materialization*"

Start from a concept forward to modify or generate structure design with high performance!

## Designs 
**Beautiful bridges:**
![](./assets/2dbridge.png)


## Update
- [x] 2D LMTO scripts
- [x] ComfyUI node and flow for LMTO

## Inference
Run optimization:
```bash
python scripts/LMTO_wandb.py
```
-  "--nelx": resolution x
-  "--nely": resolution y
-  "--rmin": minimum radius
-  "--bc": boundary condition
-  "--opt_method": optimization method
-  "--alpha": preference weighting
## Explore design space from a given image package
Sweep design domain according to a certain prompt:
```bash
wandb sweep config/sweep_bridgeForceMid.yaml
```

