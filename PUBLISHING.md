# Publish this repository to GitHub

Create an empty GitHub repository named `jax-pallas-kernel-lab`, then run from this folder:

```bash
git init
git add .
git commit -m "Initial JAX internals portfolio implementation"
git branch -M main
git remote add origin https://github.com/dennisnduto/jax-pallas-kernel-lab.git
git push -u origin main
```

Before publishing, run `pytest -q`, read every source file, and be prepared to explain the
transformation, lowering, or sharding decisions in an interview.
