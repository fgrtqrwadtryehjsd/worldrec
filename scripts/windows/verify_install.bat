@echo off
setlocal
call "E:\Soft\anaconda3\Scripts\activate.bat" worldrec
python -c "import torch; print('  torch      :', torch.__version__); print('  CUDA       :', torch.version.cuda); print('  GPU count  :', torch.cuda.device_count())" 2>nul
python -c "import torch; [print(f'  GPU {i}      :', torch.cuda.get_device_properties(i).name, f'({torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB)') for i in range(torch.cuda.device_count())]" 2>nul
python -c "import transformers; print('  transformers:', transformers.__version__)" 2>nul
python -c "import peft; print('  peft        :', peft.__version__)" 2>nul
python -c "import accelerate; print('  accelerate  :', accelerate.__version__)" 2>nul
python -c "import deepspeed; print('  deepspeed   :', deepspeed.__version__)" 2>nul
python -c "import flash_attn; print('  flash-attn  :', flash_attn.__version__)" 2>nul
endlocal
