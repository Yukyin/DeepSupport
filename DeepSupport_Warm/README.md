# DeepSupport Warm ❤️‍🩹
🫶 DeepSupport Warm is an emotional-holding companion.  
✍️ Write how you feel, then get gentle reflection and warm support, no rushing into what to do next.


![Overview](ds-warm.png)



## What it does ✨
DeepSupport Warm helps you feel held and less alone in the moment:

- 🫧 Validate and name what you’re feeling without judging.  
- 🧸 Stay with the emotion first, before problem-solving.  
- 🌙 Offer gentle grounding and a small next step only if you want it.  



## Quickstart 🚀

### 1. Dependencies
```bash
conda create -n deepsupport-warm python=3.10 -y
conda activate deepsupport-warm
cd /path/to/deepsupport-warm
pip install -r requirements.txt
```

### 2. Run with Gradio
```bash
python deepsupport_warm_demo.py \
  --lora deepsupport-warm-lora-oss \
  --host 0.0.0.0 --port 7866
```
> An open LoRA adapter is recommended and available [here](https://huggingface.co/Yukyin/deepsupport-warm-lora-oss). You can also train your own LoRA with your own data.


## Dataset and Examples 📊
- This OSS release is intended to be used with a LoRA adapter trained on de-identified versions of the original data.

- The original internal version relies on a private LoRA adapter trained on non-de-identified data and therefore cannot be open-sourced at this time. For a trial of the original version, please contact the author.

- We provide comparative examples between the OSS release and the original internal version here: **[`EXAMPLES.md`](EXAMPLES.md)**. 


## Disclaimer ⚠️
- DeepSupport Warm does not provide professional advice, diagnosis, or therapy. 

- It provides supportive conversation and emotional holding. Please seek qualified professional help when needed.


## Research and Citation 📚
If you use DeepSupport Warm in a paper, report, thesis, or study, please cite this repository:

```bibtex
@software{deepsupport_warm_2026,
  author  = {Yuyan Chen},
  title   = {DeepSupport Warm: An emotional-holding companion for supportive dialogue},
  year    = {2026},
  version = {oss},
  url     = {\url{https://github.com/Yukyin/DeepSupport/DeepSupport_Warm}}
}
```


## License 📜
- Noncommercial use (including academic and research use) is governed by `LICENSE` (PolyForm Noncommercial 1.0.0).
- Commercial use requires a separate agreement — see `COMMERCIAL_LICENSE.md`.
- Copyright notice is provided in `NOTICE`.

📨 Commercial inquiries: yolandachen0313@gmail.com
