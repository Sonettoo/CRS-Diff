# CRS-Diff: Controllable Generative Remote Sensing Foundation Model
### [Paper (ArXiv)](https://arxiv.org/abs/2403.11614) 

## Hightlights
<div align=center>
<img src="imgs/result_light.png" height="100%" width="100%"/>
</div>

## TODO

- [ ] Release training and inference code.
- [ ] Release pretrained models.
- [ ] Release Gradio UI.
- [ ] A light-weighted Latent-Composer built upon Stable Diffusion 2.1.

## Environment

```bash
conda env create -f environment.yaml
conda activate csrldm
```
You can download pre-trained models [last.ckpt](https://drive.google.com/file/d/1CJZ9CG_ssQBeww1enVoEzCyelimxPiaW/view?usp=drive_link) and put it to `./ckpt/` folder.

### Testing

You can run the code to start the gradio interface by:
```bash
python src/test/test.py
```
The demonstration effects of the project are as follows:

You can run the code to start the gradio interface by:
```bash
python src/test/test.py
```


